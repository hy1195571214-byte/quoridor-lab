import React, { useMemo, useState } from "react";
import {
  actionError,
  legalDestinations,
  notation,
  parseAction,
} from "../game/rules.js";

export default function Board({
  state,
  onAction,
  disabled = false,
  flipped = false,
  light = false,
  analysis = false,
  dragKind = null,
  dragTarget = null,
  children,
}) {
  const [hovered, setHovered] = useState(null);
  const destinations = useMemo(
    () => new Set(legalDestinations(state).map((square) => square.join(","))),
    [state],
  );
  const wallOwners = useMemo(() => {
    const owners = new Map();
    state.history.forEach((move, index) => {
      const action = parseAction(move);
      if (action.kind !== "move") owners.set(move, (index % 2) + 1);
    });
    return owners;
  }, [state.history]);
  const lastMove = state.history.at(-1);
  const cells = [];
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      const action = { kind: "move", row, col };
      const name = notation(action);
      const player =
        state.pawns.findIndex(
          (square) => square[0] === row && square[1] === col,
        ) + 1;
      const legal =
        !disabled && state.winner === null && destinations.has(`${row},${col}`);
      cells.push(
        <button
          key={name}
          className={`cell ${row === 0 ? "goal-red" : ""} ${row === 8 ? "goal-blue" : ""} ${legal && !analysis ? "legal" : ""} ${lastMove === name ? "last-move" : ""}`}
          style={{ gridRow: row * 2 + 2, gridColumn: col * 2 + 2 }}
          aria-label={`${name}${player ? (player === 1 ? "，红方棋子" : "，蓝方棋子") : ""}`}
          data-square={name}
          aria-disabled={disabled || !legal}
          onClick={() => !disabled && onAction(action)}
        />,
      );
    }
  }
  const slots = [];
  for (const kind of ["h", "v"]) {
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const action = { kind, row, col };
        const name = notation(action);
        const isHovered = dragKind ? dragTarget === name : hovered === name;
        const invalid = isHovered && actionError(state, action);
        slots.push(
          <button
            key={name}
            data-wall={name}
            data-kind={kind}
            className={`wall-slot wall-${kind} ${isHovered ? "hovered" : ""} ${invalid ? "invalid" : ""} ${dragKind && kind !== dragKind ? "other-direction" : ""}`}
            style={wallStyle(kind, row, col)}
            aria-label={`放置${kind === "h" ? "横" : "竖"}墙 ${name}`}
            disabled={disabled}
            onPointerEnter={() => setHovered(name)}
            onPointerLeave={() => setHovered(null)}
            onClick={() => !disabled && onAction(action)}
            onDragOver={(event) => {
              if (!disabled && (!dragKind || kind === dragKind)) {
                event.preventDefault();
                setHovered(name);
              }
            }}
            onDrop={(event) => {
              event.preventDefault();
              if (!disabled) onAction(action);
            }}
          />,
        );
      }
    }
  }
  return (
    <div
      className={`board ${flipped ? "flipped" : ""} ${light ? "light-board" : ""} ${analysis ? "analysis-board" : ""} ${dragKind ? "dragging" : ""}`}
      data-board-root
    >
      {cells}
      <div className="pawn-layer" aria-hidden="true">
        {state.pawns.map(([row, col], index) => (
          <span
            key={index}
            className={`pawn pawn-${index + 1}`}
            style={{
              left: `${((col + 0.5) / 9) * 100}%`,
              top: `${((row + 0.5) / 9) * 100}%`,
            }}
          />
        ))}
      </div>
      {slots}
      {state.walls.map(([kind, row, col]) => {
        const name = notation({ kind, row, col });
        return (
          <div
            key={`placed-${name}`}
            style={wallStyle(kind, row, col)}
            className={`placed-wall wall-${kind} wall-owner-${wallOwners.get(name)} ${lastMove === name ? "last-wall" : ""}`}
          />
        );
      })}
      {Array.from({ length: 9 }, (_, index) => (
        <React.Fragment key={index}>
          <span
            className="coordinate coordinate-rank"
            style={{ gridRow: index * 2 + 2, gridColumn: 19 }}
          >
            {9 - index}
          </span>
          <span
            className="coordinate coordinate-file"
            style={{ gridRow: 19, gridColumn: index * 2 + 2 }}
          >
            {String.fromCharCode(97 + index)}
          </span>
        </React.Fragment>
      ))}
      {children && <div className="board-overlay">{children}</div>}
    </div>
  );
}

function wallStyle(kind, row, col) {
  return kind === "h"
    ? { gridRow: row * 2 + 3, gridColumn: `${col * 2 + 2} / span 3` }
    : { gridRow: `${row * 2 + 2} / span 3`, gridColumn: col * 2 + 3 };
}
