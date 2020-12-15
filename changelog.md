# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `numdicts` submodule, providing dictionaries that support numerical operations and automatic differentiation.
- `utils` subpackage for miscellaneous utilities, including `pprint` submodule, which extends stdlib `pprint` to handle some `pyClarion` objects.
- Attribute `assets` for `Structure` objects. This is a simple dict for storing datastructures shared by multiple components of the parent realizer (e.g, chunk database may be shared by updaters).
- New construct types and symbols for rules, feature/chunk pools, and preprocessing flows.
- `Token` class for building structured symbolic tokens.
- Examples `lagged_features.py`, `flow_control.py`, `chunk_extraction.py`, `working_memory.py`.
- `components` submodule defining basic abstractions for defining components:
    - `Component`, `Emitter`, `Updater` abstractions for specifying components and setting up links.
    - `Propagator` and `Cycle` classes for specifying activation propagation procedures for `Construct` and `Structure` instances.
    - `Assets`, a simple namespace object for holding structure assets.
    - `FeatureDomain`, `FeatureInterface`, `SimpleDomain`, `SimpleInterface` for structuring specification of feature domains and feature driven control of components
    - `SimpleQNet` and `ReinforcementMap` for building simple Q-learning models.
- Use of `with` statements to automate adding constructs to containers.
- `AgentCycle`, `CycleS` abstraction and `ACSCycle` classes for controlling structure propagation.  
- Chunk, rule, and BLA databases `Chunks` and `Rules`, `BLAs`.
- Chunk extraction termini `ChunkExtractor` and `ControlledExtractor`.
- `Filtered`, `Gated`, and `Pruned` propagators, allowing input filtering and output gating.
- `ActionRules` propagator class.
- Buffer propagators `ParamSet`, `Register`, and `RegisterArray`.
- `blas.py` defining BLA databases and some basic related updaters.
- `updaters.py` defining updater chains and conditional updaters.

### Changed

- Reorganzied library.
- `ConstructSymbol` replaced with new `Symbol` class.
- Old construct realizer classes simplified and replaced: 
    - `Structure` class for containers
    - `Construct` class for basic constructs.
- Realizers and propagators all modified to emit and operate on numdicts, as defined by `numdicts` submodule. 
- Individual chunk and feature nodes no longer explicitly represented, instead use of feature pools is encouraged.
- `nacs_proc` function converted to `NACSCycle` class.

### Removed

- `funcs.py`
- `packets.py`

### Fixed 

- Circular imports.

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
