import os
import threading
from pathlib import Path
from typing import Callable, Iterable, Set, FrozenSet

class AuxiliaryDataRegistry:
    """
    Thread-safe registry for auxiliary dictionary sets (protected names, places,
    given names, surnames, fixed nouns, participles).
    """

    def __init__(self, load_fn: Callable, load_tagged_fn: Callable, path_resolver_fn: Callable):
        self._load_fn = load_fn
        self._load_tagged_fn = load_tagged_fn
        self._path_resolver_fn = path_resolver_fn
        self._lock = threading.Lock()

        self._protected_names: FrozenSet[str] | None = None
        self._given_names: FrozenSet[str] | None = None
        self._surnames: FrozenSet[str] | None = None

    def get_protected_names(self) -> FrozenSet[str]:
        if self._protected_names is None:
            with self._lock:
                if self._protected_names is None:
                    paths = self._path_resolver_fn("protected_names.dic", "protected_names.txt")
                    self._protected_names = self._load_fn(paths)
        return self._protected_names

    def get_given_names(self) -> FrozenSet[str]:
        if self._given_names is None:
            with self._lock:
                if self._given_names is None:
                    paths = self._path_resolver_fn("names.dic")
                    self._given_names = self._load_tagged_fn(paths, {"NAME"})
        return self._given_names

    def get_surnames(self) -> FrozenSet[str]:
        if self._surnames is None:
            with self._lock:
                if self._surnames is None:
                    paths = self._path_resolver_fn("names.dic")
                    self._surnames = self._load_tagged_fn(paths, {"SNAME"})
        return self._surnames
