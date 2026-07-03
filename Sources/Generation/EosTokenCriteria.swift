#if canImport(CoreML)
import CoreML

/// Stops generation when the most recently generated token is one of a set of
/// end-of-sequence token ids.
///
/// Mirrors Python `transformers.EosTokenCriteria`. The single `GenerationConfig.eosTokenId`
/// can only represent one end token, but many chat models declare several — e.g. Gemma 3
/// ends a turn with `<end_of_turn>` (106) while its other configured eos is `<eos>` (1).
/// Supplying those ids as a criterion lets generation stop on *any* of them.
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public struct EosTokenCriteria: StoppingCriteria {
    /// The end-of-sequence token ids; generation stops when the last token is any of these.
    public let eosTokenIds: Set<Int>

    /// Creates a criterion that stops on any of the given end-of-sequence token ids.
    public init(_ eosTokenIds: Set<Int>) {
        self.eosTokenIds = eosTokenIds
    }

    public func shouldStop(tokens: [Int], scores: MLTensor?) -> Bool {
        tokens.last.map(eosTokenIds.contains) ?? false
    }
}

#endif // canImport(CoreML)
