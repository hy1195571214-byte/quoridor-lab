import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { GlobeAltIcon } from "@heroicons/react/24/outline";
import { readPreference, writePreference } from "./game/preferences.js";
import { translate } from "./translations.js";

const LanguageContext = createContext(null);

export function LanguageProvider({ children }) {
  const [locale, setLocale] = useState(() =>
    readPreference("locale", "zh-CN") === "en" ? "en" : "zh-CN",
  );
  const t = useCallback(
    (key, values) => translate(locale, key, values),
    [locale],
  );
  useEffect(() => {
    writePreference("locale", locale);
    document.documentElement.lang = locale;
    document.title = translate(locale, "Quoridor Lab · 本地对战与回放");
  }, [locale]);
  const value = useMemo(() => ({ locale, setLocale, t }), [locale, t]);
  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}

export function LanguageSelector({ className = "", showLabel = false }) {
  const { locale, setLocale, t } = useLanguage();
  return (
    <label className={`language ${className}`}>
      <GlobeAltIcon aria-hidden="true" />
      {showLabel && <span>{t("语言")}</span>}
      <select
        aria-label={t("语言")}
        value={locale}
        onChange={(event) => setLocale(event.target.value)}
      >
        <option value="zh-CN" lang="zh-CN">
          简体中文
        </option>
        <option value="en" lang="en">
          English
        </option>
      </select>
    </label>
  );
}
