# ril-env-nightly
The nightly build has the following improvements:
- Every class is now context-managed. This is useful for
  multiprocessing.
- Everything that needs to be run now runs via python multiprocessing
  (which bypasses the GIL) and shared memory management. Easy multi"threading"
- Everything is now logged using a logger, levels are escalated using
  verbosity flags.
- The spacemouse code now uses the spnav library which makes it easier
  to define events.
- The xarm teleop now uses relative instead of absolute pose. This
  should hopefully feel more intuitive to use.
