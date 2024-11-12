import { useEffect, useState } from 'react';

export function useStickyState(defaultValue: object | string, name: string) {
  const [value, setValue] = useState(() => {
    if (typeof window === 'undefined' || !window.localStorage) {
      return defaultValue;
    }

    const persistedValue = window.localStorage.getItem(name);

    return persistedValue !== null ? JSON.parse(persistedValue) : defaultValue;
  });

  useEffect(() => {
    window.localStorage.setItem(name, JSON.stringify(value));
  }, [name, value]);

  return [value, setValue];
}
