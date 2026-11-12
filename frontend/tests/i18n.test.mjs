import test from "node:test";
import assert from "node:assert/strict";
import { englishMessages, message, translate } from "../src/translations.js";

test("English labels cover the dictionary and Chinese labels stay available", () => {
  for (const [chinese, english] of Object.entries(englishMessages)) {
    assert.ok(english.trim());
    assert.doesNotMatch(english, /[\u3400-\u9fff]/);
    assert.equal(translate("zh-CN", chinese), chinese);
    assert.equal(translate("en", chinese), english);
  }
});

test("deferred error messages use the current language and preserve details", () => {
  const error = message("棋谱无法载入。{error}", {
    error: "Invalid record JSON",
  });
  assert.equal(
    translate("en", error),
    "Could not load the game. Invalid record JSON",
  );
  assert.equal(translate("zh-CN", error), "棋谱无法载入。Invalid record JSON");
  assert.deepEqual(error.params, { error: "Invalid record JSON" });
});

test("localized parameters preserve zero values and unknown diagnostics", () => {
  assert.equal(translate("en", "{count} 步", { count: 0 }), "0 moves");
  assert.equal(
    translate("en", "放置横墙 {square}", { square: "ha8" }),
    "Place horizontal wall ha8",
  );
  assert.equal(
    translate("en", "External service error"),
    "External service error",
  );
  assert.equal(translate("en", ""), "");
});
