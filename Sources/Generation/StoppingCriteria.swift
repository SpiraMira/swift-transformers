#if canImport(CoreML)
import CoreML

/// A caller-supplied condition that can stop generation early, evaluated once per
/// generated token (after the token is appended to the sequence).
///
/// Mirrors Python `transformers.StoppingCriteria` / `StoppingCriteriaList`: a hook
/// for stopping generation on a custom condition beyond EOS / `maxLength` — e.g.
/// a deadline, a stop string, a resource budget, or any caller-defined policy.
///
/// The library makes no assumption about *why* a criterion stops; it only asks.
/// Returning `true` ends the loop and returns the partial sequence as a normal
/// (non-error) result. This is orthogonal to Swift task cancellation: cancellation
/// means "this work was abandoned" (the loop throws `CancellationError`); a stopping
/// criterion means "the caller's policy says stop here" (the loop returns cleanly).
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public protocol StoppingCriteria: Sendable {
    /// Whether generation should stop now.
    /// - Parameters:
    ///   - tokens: the full token sequence so far (prompt + completion).
    ///   - scores: the processed logits for the most recent step, if available.
    /// - Returns: `true` to stop generation.
    func shouldStop(tokens: [Int], scores: MLTensor?) -> Bool
}

#endif // canImport(CoreML)
