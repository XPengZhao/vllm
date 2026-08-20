# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming parser engine that orchestrates token ID scanning,
incremental lexing, and state-machine-driven semantic event emission."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.incremental_lexer import (
    CONTENT_TERMINAL,
    IncrementalLexer,
    LexerShape,
    LexToken,
    TerminalDef,
)
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.engine.token_id_scanner import (
    DROP_TERMINAL,
    LexerInput,
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)


@dataclass(slots=True)
class _DropInfo:
    lexer_shape: LexerShape
    extra_token_ids: dict[int, str]


def _build_drop_info(
    config: ParserEngineConfig,
    tokenizer,
) -> _DropInfo | None:
    try:
        special_tokens: list[str] = list(tokenizer.all_special_tokens)
        special_ids: list[int] = list(tokenizer.all_special_ids)
    except (AttributeError, NotImplementedError):
        return None

    if not special_tokens:
        return None

    configured_texts = (
        set(config.token_id_terminals.values())
        | set(config.terminals.values())
        | config.preserve_tokens
    )

    extra_token_ids: dict[int, str] = {}
    drop_texts: set[str] = set()
    for text, tid in zip(special_tokens, special_ids):
        if text not in configured_texts:
            extra_token_ids[tid] = DROP_TERMINAL
            drop_texts.add(text)

    if not drop_texts:
        return None

    import regex as re

    drop_terminal_defs = [
        TerminalDef(
            name=DROP_TERMINAL,
            pattern=re.compile(re.escape(text)),
            is_literal=True,
            literal=text,
        )
        for text in drop_texts
    ]

    all_terminal_defs = list(config.terminal_defs) + drop_terminal_defs
    lexer_shape = LexerShape(all_terminal_defs)

    return _DropInfo(
        lexer_shape=lexer_shape,
        extra_token_ids=extra_token_ids,
    )


class StreamingParserEngine:
    """Consumes ``(delta_text, delta_token_ids)`` pairs and produces a
    stream of :class:`SemanticEvent` instances.

    This is the main entry point for streaming parsing.
    Create one per request (it is stateful).

    The pipeline is::

        delta_text + delta_token_ids
            → TokenIDScanner  (special token pre-lexing)
            → IncrementalLexer  (text → terminal tokens with prefix buffering)
            → State Machine  (terminal → semantic events)
            → list[SemanticEvent]

    Usage::

        engine = StreamingParserEngine(config, tokenizer)
        for each streaming delta:
            events = engine.feed(delta_text, delta_token_ids)
            # convert events to DeltaMessage
    """

    def __init__(
        self,
        config: ParserEngineConfig,
        tokenizer,
        initial_state: ParserState | None = None,
        vocab: dict[str, int] | None = None,
    ) -> None:
        self.config = config

        resolved_token_ids: dict[int, str] = {}
        if tokenizer is not None:
            if vocab is None:
                vocab = tokenizer.get_vocab()
            if config.token_id_terminals:
                for terminal_name, token_text in config.token_id_terminals.items():
                    tid = vocab.get(token_text)
                    if tid is not None:
                        resolved_token_ids[tid] = terminal_name

        drop_info: _DropInfo | None = None
        if tokenizer is not None:
            drop_info = _build_drop_info(config, tokenizer)

        lexer_shape = config.lexer_shape
        if drop_info is not None:
            resolved_token_ids.update(drop_info.extra_token_ids)
            lexer_shape = drop_info.lexer_shape

        self._resolved_token_ids = resolved_token_ids
        self._has_drops = drop_info is not None

        self._scanner = TokenIDScanner(
            resolved_token_ids,
            tokenizer,
        )

        self._token_id_terminal_names: frozenset[str] = frozenset(
            resolved_token_ids.values()
        )

        self._lexer = IncrementalLexer(lexer_shape, content_terminal=CONTENT_TERMINAL)

        self._tool_terminals: frozenset[str] = frozenset(
            terminal
            for (state, terminal), tr in config.transitions.items()
            if tr.next_state in self._TOOL_STATES or state in self._TOOL_STATES
        )
        # TOOL_CALL_END may close an inner call rather than its lexical wrapper,
        # so identify exits from state transitions instead.
        self._tool_exit_terminals: frozenset[str] = frozenset(
            terminal
            for (state, terminal), tr in config.transitions.items()
            if state in self._TOOL_STATES and tr.next_state not in self._TOOL_STATES
        )

        self.skip_tool_parsing = False
        # Function names declared by the request, or None when unknown.
        # Used to abort a recovery hold early when the held name can no
        # longer become a declared tool. Survives reset() like
        # ``skip_tool_parsing``.
        self.allowed_tool_names: frozenset[str] | None = None
        # True when the request asked for tool_choice "none".
        self.suppress_tool_calls = False
        # Optional parser-owned validator used only by provisional tool-call
        # transitions. It intentionally survives reset() so the owning
        # ParserEngine can bind a stable callback once at construction.
        self.recovery_tool_name_validator: Callable[[str], bool] | None = None
        self.reset(initial_state=initial_state)

    def _reset_args_state(self) -> None:
        self._args_buffer: str = ""
        self._args_safe_end: int = 0
        self._args_brace_depth: int = 0
        self._args_in_string: bool = False
        self._args_escape_next: bool = False

    def reset(self, initial_state: ParserState | None = None) -> None:
        """Reset mutable state for reuse across requests.

        Preserves cached immutable structures (compiled terminals,
        resolved token IDs, lexer shape, token text cache) to avoid
        redundant initialization work.
        """
        self.state = (
            initial_state if initial_state is not None else self.config.initial_state
        )
        self.tool_index = -1
        self._ever_had_token_ids = False
        # DO NOT reset skip_tool_parsing here — callers set it before
        # calling methods that trigger reset() (e.g. extract_reasoning),
        # and clearing it silently breaks non-streaming tool-call-as-
        # implicit-reasoning-end (content returns None).
        self._scanner.reset()
        self._lexer.reset()
        self._message_header_buffer = ""
        self._reset_args_state()
        self._recovery_hold_active = False
        self._recovery_hold_events: list[SemanticEvent] = []
        self._recovery_hold_raw = ""
        self._recovery_hold_name = ""
        self._recovery_prior_state = self.state
        self._recovery_prior_tool_index = -1
        self._recovery_outer_closer_pending = False

    def feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[SemanticEvent]:
        if delta_token_ids:
            self._ever_had_token_ids = True

        # Fast path: skip scanner and lexer when the delta is plain
        # content with no special tokens and no terminal-starting chars.
        if (
            delta_text
            and not self._lexer.buffer
            and not self._scanner._deferred_terminals
            and self._lexer._literal_first_chars.isdisjoint(delta_text)
        ):
            has_special = False
            for tid in delta_token_ids:
                if tid in self._resolved_token_ids:
                    has_special = True
                    break
            if not has_special:
                return self._emit_for_state(delta_text)

        scanner_items = self._scanner.scan(delta_text, delta_token_ids)

        if len(scanner_items) == 1 and isinstance(scanner_items[0], TextChunk):
            lex_tokens = self._lexer.feed(scanner_items[0].text)
            if len(lex_tokens) == 1 and lex_tokens[0].terminal == CONTENT_TERMINAL:
                text = lex_tokens[0].value
                return self._emit_for_state(text)
            return self._process_lex_tokens(lex_tokens)

        return self._process_scanner_items(scanner_items)

    def _process_scanner_items(
        self, items: Sequence[LexerInput]
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        for item in items:
            if isinstance(item, PreLexedTerminal):
                events.extend(self._process_lex_tokens(self._lexer.flush()))
                events.extend(self._on_terminal(item.terminal, item.text))
            elif isinstance(item, TextChunk):
                events.extend(self._process_lex_tokens(self._lexer.feed(item.text)))
        return events

    def finish(self) -> list[SemanticEvent]:
        events = self._process_scanner_items(self._scanner.flush_pending())

        events.extend(self._process_lex_tokens(self._lexer.flush()))

        if self._recovery_hold_active:
            events.extend(self._abort_recovery_hold())

        if self._args_buffer:
            events.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=self._args_buffer,
                    tool_index=self.tool_index,
                )
            )
            self._args_buffer = ""
            self._args_safe_end = 0

        if self.state in (
            ParserState.TOOL_PREAMBLE,
            ParserState.TOOL_ARGS,
            ParserState.TOOL_NAME,
            ParserState.TOOL_BETWEEN,
        ):
            if self.tool_index >= 0:
                events.append(
                    SemanticEvent(
                        EventType.TOOL_CALL_END,
                        tool_index=self.tool_index,
                    )
                )
            self.state = ParserState.CONTENT
        elif self.state == ParserState.REASONING:
            events.append(
                SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index)
            )
            self.state = ParserState.CONTENT
        elif self.state == ParserState.MESSAGE_HEADER:
            if self._message_header_buffer:
                events.append(
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=self._message_header_buffer,
                        tool_index=self.tool_index,
                    )
                )
                self._message_header_buffer = ""
            self.state = ParserState.CONTENT

        return events

    def parse_complete(self, text: str) -> list[SemanticEvent]:
        token_ids: list[int] = []
        events = self.feed(text, token_ids)
        events.extend(self.finish())
        return events

    def _process_lex_tokens(self, tokens: list[LexToken]) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        strict = self._token_id_terminal_names if self._ever_had_token_ids else None
        for tok in tokens:
            if tok.terminal == CONTENT_TERMINAL or (strict and tok.terminal in strict):
                events.extend(self._on_content(tok.value))
            else:
                events.extend(self._on_terminal(tok.terminal, tok.value))
        return events

    _TOOL_STATES = frozenset(
        {
            ParserState.TOOL_PREAMBLE,
            ParserState.TOOL_NAME,
            ParserState.TOOL_ARGS,
            ParserState.TOOL_BETWEEN,
        }
    )

    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:
        if (
            self._recovery_outer_closer_pending
            and self.state == ParserState.CONTENT
            and terminal in self._tool_exit_terminals
        ):
            self._recovery_outer_closer_pending = False
            return []

        key = (self.state, terminal)
        transition = self.config.transitions.get(key)

        if transition is None:
            # DROP_TERMINAL must never enter provisional recovery buffers.
            if self._has_drops and terminal == DROP_TERMINAL:
                return []
            if self._recovery_hold_active and self.state == ParserState.TOOL_NAME:
                events = self._abort_recovery_hold()
                events.extend(self._on_terminal(terminal, value))
                return events
            if self._recovery_hold_active and self.state == ParserState.TOOL_ARGS:
                # DSML parameter closers are terminals but deliberately have
                # no state transition. During recovery they are still part of
                # the candidate invoke and must stay buffered as arguments.
                return self._emit_for_state(value)
            return self._emit_for_state(value)

        if self.skip_tool_parsing and (
            transition.provisional_tool_call or self._recovery_hold_active
        ):
            return self._apply_transition(transition, value)

        if self.skip_tool_parsing and terminal in self._tool_terminals:
            if self.state == ParserState.MESSAGE_HEADER:
                self.state = ParserState.CONTENT
                self._message_header_buffer = ""
                return [
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=value,
                        tool_index=self.tool_index,
                    )
                ]
            if EventType.REASONING_END in transition.events:
                self.state = ParserState.CONTENT
                return [
                    SemanticEvent(
                        EventType.REASONING_END,
                        value=value,
                        tool_index=self.tool_index,
                    ),
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=value,
                        tool_index=self.tool_index,
                    ),
                ]
            content_type = self.config.content_events.get(self.state)
            if content_type is not None:
                return [
                    SemanticEvent(content_type, value=value, tool_index=self.tool_index)
                ]
            return []

        if transition.skip_in_token_id_mode and self._ever_had_token_ids:
            return self._emit_for_state(value)

        return self._apply_transition(transition, value)

    def _emit_for_state(self, text: str) -> list[SemanticEvent]:
        if self._recovery_hold_active:
            return self._hold_recovery_text(text)
        return self._emit_for_state_now(text)

    def _emit_for_state_now(self, text: str) -> list[SemanticEvent]:
        if self.state == ParserState.MESSAGE_HEADER:
            self._message_header_buffer += text
            return []
        if self.state == ParserState.TOOL_ARGS:
            if self.config.tool_args_json:
                return self._feed_args_text(text)
            return [
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=text,
                    tool_index=self.tool_index,
                )
            ]
        content_type = self.config.content_events.get(self.state)
        if content_type is not None:
            return [SemanticEvent(content_type, value=text, tool_index=self.tool_index)]
        return []

    def _hold_recovery_text(self, text: str) -> list[SemanticEvent]:
        self._recovery_hold_raw += text

        if self.state == ParserState.TOOL_NAME:
            self._recovery_hold_name += text
            # Tool names are expected to be compact. Avoid delaying an entire
            # response if ordinary prose happens to quote an unterminated
            # invoke marker.
            if len(self._recovery_hold_name) > 256 or "\n" in self._recovery_hold_name:
                return self._abort_recovery_hold()
            if not self._can_grow_into_declared_name(self._recovery_hold_name):
                return self._abort_recovery_hold()

        self._recovery_hold_events.extend(self._emit_for_state_now(text))
        return []

    def _on_content(self, text: str) -> list[SemanticEvent]:
        if not text:
            return []
        return self._emit_for_state(text)

    def _apply_transition(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        if self._recovery_hold_active:
            return self._advance_recovery_hold(transition, value)
        if transition.provisional_tool_call:
            if self.recovery_tool_name_validator is None:
                return self._emit_for_state(value)
            return self._begin_recovery_hold(transition, value)
        if (
            self._recovery_outer_closer_pending
            and self.state == ParserState.CONTENT
            and transition.next_state in self._TOOL_STATES
        ):
            self._recovery_outer_closer_pending = False
        return self._run_transition(transition, value)

    def _begin_recovery_hold(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        prior_state = self.state
        prior_tool_index = self.tool_index
        held_events = self._run_transition(transition, value)
        self._recovery_hold_active = True
        self._recovery_hold_events = held_events
        self._recovery_hold_raw = value
        self._recovery_hold_name = ""
        self._recovery_prior_state = prior_state
        self._recovery_prior_tool_index = prior_tool_index
        return []

    def _advance_recovery_hold(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        self._recovery_hold_raw += value

        if self.state == ParserState.TOOL_NAME:
            validator = self.recovery_tool_name_validator
            if validator is None or not validator(self._recovery_hold_name):
                return self._abort_recovery_hold()
            self._recovery_hold_events.extend(self._run_transition(transition, value))
            return []

        if self.state == ParserState.TOOL_ARGS:
            if not transition.commit_provisional_tool_call:
                return self._abort_recovery_hold()
            if self.skip_tool_parsing:
                raw = self._recovery_hold_raw
                prior_state = self._recovery_prior_state
                prior_tool_index = self._recovery_prior_tool_index
                self.state = ParserState.CONTENT
                self.tool_index = prior_tool_index
                self._reset_args_state()
                self._clear_recovery_hold()
                self._recovery_outer_closer_pending = True
                events: list[SemanticEvent] = []
                if prior_state == ParserState.REASONING:
                    events.append(
                        SemanticEvent(
                            EventType.REASONING_END,
                            tool_index=prior_tool_index,
                        )
                    )
                events.append(
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=raw,
                        tool_index=prior_tool_index,
                    )
                )
                return events
            transition_events = self._run_transition(transition, value)
            self._recovery_hold_events.extend(transition_events)
            events = self._recovery_hold_events
            self._clear_recovery_hold()
            # A recovered invoke started outside a valid outer tool wrapper.
            # Do not leave it in TOOL_BETWEEN: subsequent text is content, and
            # a following bare invoke must enter the provisional path again.
            self.state = ParserState.CONTENT
            self._recovery_outer_closer_pending = True
            return events

        return self._abort_recovery_hold()

    def _abort_recovery_hold(self) -> list[SemanticEvent]:
        raw = self._recovery_hold_raw
        self.state = self._recovery_prior_state
        self.tool_index = self._recovery_prior_tool_index
        self._reset_args_state()
        self._clear_recovery_hold()
        return self._emit_for_state_now(raw)

    def _clear_recovery_hold(self) -> None:
        self._recovery_hold_active = False
        self._recovery_hold_events = []
        self._recovery_hold_raw = ""
        self._recovery_hold_name = ""

    def _can_grow_into_declared_name(self, candidate: str) -> bool:
        """Return True when *candidate* is a prefix of a declared tool name.

        Consulted while a recovery hold is active. Once the text seen so
        far stops being a prefix of any declared name the caller aborts
        the hold so quoted markers stream out as content promptly.
        """
        allowed = self.allowed_tool_names
        if allowed is None:
            return True
        return any(name.startswith(candidate) for name in allowed)

    def _run_transition(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        previous_state = self.state
        message_header = ""

        if (
            self.state == ParserState.TOOL_ARGS
            and transition.next_state != ParserState.TOOL_ARGS
            and self._args_buffer
        ):
            events.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=self._args_buffer,
                    tool_index=self.tool_index,
                )
            )
            self._args_buffer = ""

        if previous_state == ParserState.MESSAGE_HEADER:
            message_header = self._message_header_buffer
            self._message_header_buffer = ""

        self.state = transition.next_state

        for event_type in transition.events:
            if event_type == EventType.TOOL_CALL_START:
                self.tool_index += 1
            event_value = (
                message_header
                if previous_state == ParserState.MESSAGE_HEADER
                and event_type == EventType.TEXT_CHUNK
                else value
            )
            if event_type == EventType.TEXT_CHUNK and not event_value:
                continue
            events.append(
                SemanticEvent(
                    event_type,
                    value=event_value,
                    tool_index=self.tool_index,
                )
            )

        if self.state == ParserState.TOOL_ARGS:
            self._args_brace_depth = 0
            self._args_in_string = False
            self._args_escape_next = False
            self._args_safe_end = 0

        return events

    def _feed_args_text(self, text: str) -> list[SemanticEvent]:
        """Feed text into the JSON argument streaming buffer.

        Streams argument characters incrementally while holding back
        closing braces/brackets that might change as more input arrives.
        """
        events: list[SemanticEvent] = []
        for ch in text:
            result = self._feed_args_char(ch)
            events.extend(result)
        return events

    def _feed_args_char(self, ch: str) -> list[SemanticEvent]:
        self._args_buffer += ch

        if self._args_escape_next:
            self._args_escape_next = False
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if self._args_in_string:
            if ch == "\\":
                self._args_escape_next = True
            elif ch == '"':
                self._args_in_string = False
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch == '"':
            self._args_in_string = True
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch in ("{", "["):
            self._args_brace_depth += 1
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch in ("}", "]"):
            if self._args_brace_depth > 0:
                self._args_brace_depth -= 1
            if self._args_brace_depth == 0:
                return []
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        self._args_safe_end = len(self._args_buffer)
        return self._flush_safe_args()

    def _flush_safe_args(self) -> list[SemanticEvent]:
        """Emit buffered argument characters up to the safe-end watermark.

        Top-level closing braces are held back (safe_end not advanced)
        until confirmed safe by a subsequent character or finish().
        """
        if self._args_safe_end == 0:
            return []
        to_emit = self._args_buffer[: self._args_safe_end]
        self._args_buffer = self._args_buffer[self._args_safe_end :]
        self._args_safe_end = 0
        return [
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                value=to_emit,
                tool_index=self.tool_index,
            )
        ]
