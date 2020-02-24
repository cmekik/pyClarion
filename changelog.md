# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `Map2VectorEncoder`, which encodes activation maps as vectors for processing by neural nets.
- `FilteredProc`, allowing input/output filtering and source filtering.
- Attribute `shared` for container construct realizers. This is a simple dict for storing datastructures shared by multiple components of the parent realizer (e.g, chunk database may be shared by updaters).
- `SubsystemPacket` class for reporting subsystem states.
- Added `output_value` attribute to node realizers for easy inspection of current activation.
- Added `ConstructType.from_str()`.
- Added `Packet.pstr()` returning nicely formatted string representations for reporting and inspection.
- Added abstract `Proc` class for specifying propagation (i.e., forward-pass) procedures to basic construct realizers.
- Added an `options` parameter to `ConstructRealizer.propagate()` allowing calls to `propagate()` to modify construct behavior. Extended `BasicConstructRealizer.proc` callback attributes to accept `**kwargs`. 
- Added several new construct realizer constructors (e.g., `FlowTT`) for convenient initialization.
- `FeatureSymbol` subclass of `ConstructSymbol` allows direct access to `dim` 
and `val` attributes.
- `ContainerConstructRealizer.make_links()` to set up construct links after pull rules have been specified.
- `matches` attribute to `BasicConstructRealizer` and `SubsystemRealizer` for customizing how construct realizers connect to each other.

### Changed

- `nacs_proc` function converted to `NACSProc` class.
- `funcs.py` moved into newly minted `utils` subpackage.
- Construct realizers may be associated with multiple updaters instead of just one. Updaters are called in order of insertion (uses ordered dicts to be safe).
- Subsystems emit and buffers expect to (optionally) receive `SubsystemPacket` objects.
- Renamed construct realizers to have more succinct names.
- `ConstructSymbol` now accepts strings and ints to `ctype` argument (values 
must represent valid ctype).
- `ConstructSymbol` reprs now return runnable code which will reproduce the same construct symbol. 
- Simplified construct realizer class names and initalization.
- `ConstructSymbol` rewritten; now easily extensible.
- `may_connect()` removed from subsystem and agent realizers, connection 
decisions devolved to receivers through `realizer.accepts()`.

## 0.13.1 (2019-03-07)

### Added

- `SimpleFilterJunction` for filtering flow/response inputs. 
- `ConstructType.NullConstruct` alias for empty flag value.

### Changed

- Python version requirement dropped down to `>=3.6`.
- Reworked `examples/free_association.py` to be more detailed and more clear.
- Replaced all `is` checks on flags and construct symbols with `==` checks.

### Fixed

- `ConstructRealizer` could not be initialized due to failing construct symbol check and botched `__slots__` configuration.
- `BasicConstructRealizer.clear_activations()` may throw attribute errors when realizer has already been cleared or has no output.
- `SimpleNodeJunction` would not recognize a construct symbol of the same form as its stored construct symbol. This caused nodes to fail to output activations. Due to use of `is` in construct symbol checks (should have used `==`).

## 0.13.0

### Added

- `CategoricalSelector` object for categorical choices (essentially a boltzmann selector but on log strengths).
- `ConstructRealizer.clear_activations()` for resetting agent/construct output state.
- multiindexing for `ContainerConstructRealizer` members.
- functions `make_realizer`, `make_subsystem`, `make_agent` for initializing empty realizers from symbolic templates.
- `ConstructRealizer.ready()` method signaling whether realizer initalization is complete.
- `ConstructRealizer.missing()` and `ContainerConstructRealizer.missing_recursive()` for identifying missing realizer components.
- `SubsystemRealizer` and `AgentRealizer` automatically connect member realizers 
upon insertion (using data present in members' construct symbols) and disconnect
them upon deletion. 
- `BufferRealizer.propagate()` docs now contain a warning about potential unexpected/unwanted behavior.

### Changed

- `Microfeature` renamed to `Feature` for brevity and to better reflect theory.
- `ContainerConstructRealizer` properties now return lists instead of iterables for easier interactive inspection. Generators still acssessible through iterator methods such as `realizer.iter_ctype()` and `realizer.items_ctype()`.
- Improved `str` and `repr` outputs for construct symbols and realizers.
- `SimpleBoltzmannSelector` renamed `BoltzmannSelector`
- Additional `ConstructRealizer` subclass initialization arguments now optional.
- `Behavior`, `Buffer` and `Response` factories assign `BehaviorID`, 
`BufferID`, and `ResponseID` tuples as construct identifiers.
- `Appraisal` construct renamed `Response` to avoid association with appraisal theory.

### Fixed

- Simplified `BasicConstructRealizer` initialization and data model.
- Bug in `SubsystemRealizer` allowing connections between constructs that should 
not be linked.
- Bug in `ConstantSource` allowing mutation of output activation packets. 

### Removed

- Dependency on `numpy`.
