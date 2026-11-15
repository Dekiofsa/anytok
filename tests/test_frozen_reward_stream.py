import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from frozen_training_reward import FreshRewardStream


def test_long_reserved_bank_is_skipped_without_aborting_or_scoring_it():
    class Corpus:
        epochs={'web-en':0}
        calls=0
        def next_text(self,name,chars):
            self.calls+=1
            return 'reserved' if self.calls<=5000 else 'fresh'
    stream=FreshRewardStream.__new__(FreshRewardStream)
    stream.corpus=Corpus();stream.reserved={hashlib.sha256(b'reserved').hexdigest()};stream.reserved_skipped=0
    assert stream.raw('web-en')=='fresh'
    assert stream.reserved_skipped==5000 and stream.corpus.calls==5001


def test_completely_reserved_bucket_fails_instead_of_looping_forever():
    class Corpus:
        epochs={'web-en':0}
        def next_text(self,name,chars):
            self.epochs[name]+=1
            return 'reserved'
    stream=FreshRewardStream.__new__(FreshRewardStream)
    stream.corpus=Corpus();stream.reserved={hashlib.sha256(b'reserved').hexdigest()};stream.reserved_skipped=0
    with pytest.raises(ValueError,match='No unreserved'):
        stream.raw('web-en')
