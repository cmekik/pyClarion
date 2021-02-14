# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `Domain.disjoint(*domains)` for checking if domains are mutually disjoint.
- `Chunks.support(self, *domains)` to check if domains support cdb instance self.
- `Rules.support(self, *cdbs)` to check if cdbs support rdb instance self.

### Changed

- Adjusted defaults for buffer and goal stay interfaces.

## [v0.17.0] (2021-01-28)

### Added

- `BLAs.prune()`, which removes BLA records below threshold.
- Automatic symbolic address expansion.

### Changed

- `CompositeProcess` renamed to `Composite`
- `WrappedProcess` renamed to `Wrapped`
- `FeatureDomain` and `SimpleDomain` replaced with `Domain`
- `FeatureInterface` and `SimpleInterface` replaced with `Interface`
- `ReinforcementDomain` renamed to `Reinforcements`
- Simplified implementation and control interfaces for `ParamSet`, `Register`, `RegisterArray`, `GoalStay`
- `Process` input types now `Mapping[Any, NumDict]`. This is a compromise. The ideal would be to set keys to be of type `SymbolicAddress`. However, `SymbolicAddress` is a union type and `Mapping` is invariant in its key type, therefore being explicit about the key type is prone to false alarms.

### Removed

- `ControlledExtractor`

## [v0.16.0] (2021-01-17)

### Added

- Optional `blas` argument to `Register` and `RegisterArray`, supporting BLA based deletion of stored entries.
- New construct type `updater` for constructs solely dedicated to update processes (e.g., for databases, or due to sequencing requirements). 
- Automatic activation sequence generation at assembly time: `Structure` instances now step member constructs in roughly the order they were added to the structure. 
- `CompositeProcess` and `WrappedProcess` classes for simplifying compositional component definitions.
- Construct input structure checking and automated input extraction (see `Process.check_inputs()` and `Process.extract_inputs()`).

### Changed

- Constructs now connect directly to each other.
- Replaced `SymbolTries` in inputs and outputs with flat mappings from symbolic addresses to numdicts.
- Renamed `BLADrivenStrengths` to `BLAStrengths`
- Combined `BLAInvocationTracker` and `BLADrivenDeleter` into `BLAMaintainer`
- Renamed `ReniforcementMap` to `ReinforcementDomain`.
- Replaced `Component`, `Emitter`, `Propagator` with `Process`.
- Former `Updater` components recast as `Process` components serving `updater` constructs.
- Realizers now structurally immutable (see removed for list of removed methods.). Behavior can still be modified by replacing emitters, but constructs may not be added or removed after initial assembly.
- Streamlined realizer construction; use of with syntax encouraged exclusively:
    - Several `Realizer` methods now protected: `offer()`, `accepts()`, `finalize_assembly()`
    - Several `Structure` methods now protected: `add()`, `update_links()`, `offer()`, `finalize_assembly()`

### Removed 

- `SymbolTrie`
- `RegisterArrayBLAUpdater`, added BLA support to `Register` and `RegisterArray` instead (see added)
- `Updater` and all child abstractions.
- `Cycle` and all child classes and submodule `components.cycles`.
- Separate update cycle; stepping constructs now issues calls to a single stepping function only (`step()` for basic constructs).
- Removed realizer mutation methods:
    - Several `Realizer` methods: `drop()`, `clear_inputs()`
    - Several `Structure` methods: `__delitem__()`, `remove()`, `clear()`, `drop()`, `clear_inputs()`, `clear_links()`, `reweave()`

## [0.15.0] (2020-12-24)

### Added

- Runtime checkable generic protocol `SymbolTrie`
- New differentiable op `tanh` in `numdicts.ops`.
- Functions `squeeze`, `with_default`, `val_max`, `val_min`, `all_val`, `any_val` in `numdicts.funcs`
- Compact rule definitions; enabled by nested use of `Chunks.define` and 
`Rules.define` (see Changed).
- Delete requests for Chunk, Rule, and BLA databases.
- `GoalStay` propagator for maintaining and coordinating goals.
- `BLADrivenStrengths` propagator for determining chunk (and other construct) strengths based on their BLAs.
- `MSCycle`, experimental activation cycle for motivational subsystem

### Changed

- Type `Inputs` to `SymbolTrie[NumDict]` to be more precise.
- `Chunks.link()` renamed `Chunks.define()` and returns a `chunk`.
- `Rules.link()` renamed `Rules.define()` and returns a `rule`.
- For `Chunks` and `Rules`: `request_update` renamed to `request_add`, `resolve_update_requests` renamed `step`
- `BLAs.update()` renamed `BLAs.step()`

### Fixed

- `Structure` output type.
- `PullFunc` output type.
- `PullFuncs` output type.
- Incorrect filtering behaviour for `MutableNumDict.keep()` and `MutableNumDict.drop()`. 
- `Pruned.preprocess()`

### Removed

- `BLAs.request_reset()`

## [0.14.0] (2020-12-16)

### Added

- Several additions to `components` subpackage, including:
    - `chunks_` submodule defining chunk databases and several related components.
    - `rules` submodule defining rule databases and several related components.
    - `qnets` submodule defining `SimpleQNet` and `ReinforcementMap` for building simple Q-learning models.
    - `buffers` submodule defnining buffer propagators `ParamSet`, `Register`, and `RegisterArray`.
    - `blas` defining BLA databases and some basic related updaters.
- Various quality of life improvements including:
    - Use of `with` statements to automate adding constructs to containers.
    - `assets` attribute for `Structure` objects for storing datastructures shared by multiple components of the parent realizer (e.g, chunk database may be shared by updaters).
    - `utils.pprint` submodule which extends stdlib `pprint` to handle some `pyClarion` objects.
- Examples `flow_control.py`, `q_learning.py`, `chunk_extraction.py`, `working_memory.py`.
- New construct types and symbols for rules, feature/chunk pools, and preprocessing flows.
- `numdicts` subpackage, providing dictionaries that support numerical operations and automatic differentiation.
- `base.components` submodule defining basic abstractions for components:
    - `Component`, `Emitter`, `Updater` abstractions for specifying components and setting up links.
    - `Propagator` and `Cycle` classes for specifying activation propagation procedures for `Construct` and `Structure` instances.
    - `Assets`, a simple namespace object for holding structure assets.
    - `FeatureDomain`, `FeatureInterface`, `SimpleDomain`, `SimpleInterface` for structuring specification of feature domains and feature driven control of components

### Changed

- Reorganzied library. The basic design has persisted, but almost everything else has changed. Expect no backwards compatibility. Some notable changes include:
    - `ConstructSymbol` replaced with new `Symbol` class.
    - Old construct realizer classes simplified and replaced: 
        - `Structure` class for containers
        - `Construct` class for basic constructs.
    - Realizers and propagators all modified to emit and operate on numdicts, as defined by `numdicts` submodule. 
    - Individual chunk and feature nodes no longer explicitly represented, instead use of feature pools is encouraged.

### Fixed 

- Circular imports.

## [0.13.1] (2019-03-07)

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

## [0.13.0]

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
