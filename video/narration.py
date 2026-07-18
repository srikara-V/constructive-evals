"""Narration script for the Khan Academy-style methodology video.

Each scene is narrated continuously; visual events in scenes.py are anchored
to exact substrings of these texts and timed via ElevenLabs character-level
alignment. All numbers are taken from the experiments in this repository:

  - experiments/persona_linkedin/results.txt   (62.3% -> 88.1%, 318 pairs,
    200 extraction samples -> 161 correct / 39 incorrect, separation 3.05,
    flips +95/-13, Qwen2.5-3B-Instruct, layer -8, beta=1.0)
  - experiments/persona_tweet_virality/results.json  (0.565 baseline,
    0.575 majority-of-8, 0.58 persona-of-8)
"""

SCENES = [
    {
        "id": "01_problem",
        "title": "Ranking content: which one wins?",
        "text": (
            "Say you've got two posts sitting in front of you. Same platform, same day. "
            "One of them is going to take off, with thousands of reactions, and the other one "
            "is going to sit there quietly with almost none. Here's the question this whole "
            "video is about: can we build a system that looks at two pieces of content, "
            "two LinkedIn posts, two tweets, two images, and tells us, reliably, which one will win? "
            "The setup we'll use all the way through is called pairwise ranking. Instead of asking "
            "how many likes something will get, which is brutally hard, we just ask: A or B? "
            "Which one gets more engagement? And if you can answer that better than a coin flip, "
            "you can rank anything: posts, thumbnails, whole content libraries, because a full "
            "ranking is just many pairwise comparisons stitched together. So today I want to build up, "
            "from first principles, the method that took us from near coin-flip performance to "
            "almost ninety percent on this problem. And the way we'll get there is by looking "
            "inside a language model's head."
        ),
    },
    {
        "id": "02_just_ask",
        "title": "Why not just ask the model?",
        "text": (
            "Now, the obvious first move, the thing everybody tries, is to just ask a language model. "
            "We take a small open model, this is Qwen two point five, three billion parameters, "
            "we paste in both posts, and we ask: which of these got more engagement? Answer A or B. "
            "And the model answers. The problem is how often that answer is right. "
            "On our LinkedIn benchmark, three hundred and eighteen held-out pairs, asking directly "
            "gets about sixty-two percent. And on pairs of tweets from the same author, it's even "
            "uglier: around fifty-six percent, barely above the coin flip. "
            "So is the model just clueless about what makes content work? Well, here's the thing, "
            "and this is really the founding intuition for everything that follows: a model can know "
            "more than it says. That final printed token, A or B, is a one-bit summary of an enormous "
            "internal computation. If we only read the output, we are throwing almost all of that "
            "computation away. So here's the plan: stop treating the model as a black box that prints "
            "letters, and start reading the state of the computation itself."
        ),
    },
    {
        "id": "03_activations",
        "title": "Activations: the state inside",
        "text": (
            "So let's open the box. You already know the rough shape of a transformer: the prompt "
            "gets chopped into tokens, every token becomes a vector, and those vectors flow up "
            "through a stack of layers; this little model has thirty-six of them. "
            "Here's the part that matters for us. Between the layers, there's a running vector for "
            "every token. People in interpretability call this the residual stream. Each layer reads it, "
            "computes something, and adds its result back in. So at every layer, for every token, "
            "there is a big vector, about two thousand numbers in this model, that is, literally, "
            "the model's working memory at that point in the computation. These vectors are called "
            "activations. And the beautiful thing is: we can just record them. We run the model, "
            "we put a little hook at one layer, we used the eighth layer from the end, and while "
            "the model writes its answer, we save the activation at every response token. "
            "Then we average them, that's called mean pooling, so that every answer the model writes "
            "gets summarized as one single vector, h. Think of h as a snapshot of the model's "
            "state of mind while it was answering."
        ),
    },
    {
        "id": "04_contrastive",
        "title": "The contrastive vector",
        "text": (
            "Now for the core trick of this whole method, and it comes straight out of "
            "interpretability research. It's called a contrastive vector; you'll also hear "
            "steering vector, or persona vector. Here's the recipe. We take the training set and let "
            "the model answer a couple hundred of these A-or-B questions. For each one, we keep that "
            "snapshot vector h, and, because this is training data, we also know whether the model's "
            "answer was actually right. So now we have two piles of snapshots. Green ones, where the "
            "model happened to be right; in our LinkedIn run that was a hundred and sixty-one of them. "
            "And red ones, where it was wrong; that was thirty-nine. "
            "Then we do the simplest thing you could possibly do. We average the green pile to get one "
            "point, mu correct. We average the red pile to get mu wrong. And we subtract: "
            "v equals mu correct minus mu wrong, and then we scale v to length one. "
            "Why does subtracting means make sense? Because everything the two piles share, the topic, "
            "the format, the language, shows up in both averages, and cancels out. Whatever survives "
            "the subtraction is precisely what is different about the model's internal state when it "
            "is being right versus when it is being wrong. One direction, in activation space, that "
            "points from confused toward clear-headed. And that separation is not hypothetical: "
            "when we project both piles onto v, the green cloud and the red cloud land about three "
            "units apart in our run."
        ),
    },
    {
        "id": "05_projection",
        "title": "Scoring answers by projection",
        "text": (
            "So what do we actually do with this direction v? We use it as a measuring stick. "
            "Take any new answer the model writes, and remember, at test time we don't know if it's "
            "right, that's the whole problem. Grab its snapshot h, and compute one dot product: "
            "s equals h dot v. Geometrically, that is just dropping the shadow of h onto the "
            "direction v; a projection. If the shadow lands far along v, then the model's internal "
            "state looks like the states it had when it was answering correctly. If it lands low, "
            "it looks like the wrong-answer states. One dot product, a couple thousand "
            "multiplications, and we get a number that behaves like a confidence meter we never "
            "trained. No extra model, no labels at test time, no fine-tuning. In interpretability "
            "terms, v is acting as a linear probe. And once every answer comes with a score, "
            "a whole world opens up, because now we can compare answers."
        ),
    },
    {
        "id": "06_best_of_n",
        "title": "Sampling: best-of-N with an internal judge",
        "text": (
            "Here is the simplest way to cash that in. At test time, don't generate one answer. "
            "Turn up the temperature and sample several, say eight of them. Because sampling is "
            "random, some of those answers will say A and some will say B. And, more importantly, "
            "some of them were produced while the model was in a clear-headed state, and some were not. "
            "So: score every sample with our measuring stick, and keep the answer whose internal "
            "state scored highest. That's the whole algorithm: best-of-N with an internal judge. "
            "Now compare that to majority voting, which also samples eight times, but just counts "
            "the letters. Voting treats every sample as equally trustworthy. Our score does not; "
            "it weighs each sample by what the model's activations looked like while producing it. "
            "On the tweet benchmark, this gives a real but honestly modest bump: fifty-six and a "
            "half percent baseline, fifty-seven and a half for majority voting with eight samples, "
            "fifty-eight for persona scoring with eight samples. Same-author tweet pairs are "
            "genuinely noisy; sometimes the same person posts the same joke twice and one copy "
            "simply dies. But the machinery works. And on cleaner data, it is about to pay off "
            "much, much bigger."
        ),
    },
    {
        "id": "07_power_mh",
        "title": "Power persona sampling",
        "text": (
            "Now for the full machinery, the version that produced the big number. It's called "
            "power persona sampling, and it upgrades best-of-N in one deep way: instead of sampling "
            "answers independently and hoping one of them lands in a good state, we go searching "
            "for that state. First, we let the model reason. Instead of demanding a bare A or B, "
            "we ask for a short chain of thought and then the answer, so each attempt is now a whole "
            "reasoning trajectory. Then we run a search over trajectories called Metropolis Hastings, "
            "the classic Markov chain Monte Carlo algorithm. It works like this. You keep one current "
            "trajectory. To propose a new one, pick a random cut point in the reasoning, keep "
            "everything before the cut, and let the model resample everything after it. "
            "Then you score both trajectories: the persona score from the activations, and the "
            "model's own log probability of the text. And you accept or reject the proposal "
            "stochastically, using this rule: the log acceptance ratio equals alpha minus one, "
            "times the change in log probability, plus beta, times the change in persona score. "
            "If the proposal improves things, you tend to keep it. If it's worse, you sometimes keep "
            "it anyway, and that bit of randomness is what keeps the search from getting stuck in a rut. "
            "Run ten or twenty steps of this, and the chain drifts into reasoning that the model "
            "itself considers fluent, and that sits deep in the clear-headed region of activation "
            "space. In distribution terms, we are sampling text with probability proportional to the "
            "model's own distribution, raised to the power alpha, times e to the beta times the "
            "persona score. That power is where the name comes from, and beta is the knob for how "
            "hard we push toward the good region. And on LinkedIn, this is dramatic. Baseline: "
            "sixty-two point three percent. Power persona sampling with beta equal to one: "
            "eighty-eight point one percent. Of the pairs where it changed its mind, ninety-five "
            "flipped from wrong to right, and only thirteen flipped the other way. Same model, "
            "same weights, nothing was trained. We just used the model's own internal signal "
            "to steer its own sampling."
        ),
    },
    {
        "id": "08_images",
        "title": "From text to images (and video)",
        "text": (
            "So we built all of this on text. Now here's the payoff question: what did any of it "
            "actually assume about text? Look back at the recipe. Record activations; those are just "
            "vectors between layers. Build a contrastive direction; that's subtracting two group "
            "averages. Score by projection; that's a dot product. Search with that score. "
            "Nothing in there cares what the input tokens were before they became vectors. "
            "So let's swap the front end. A vision language model, like the Gemma family we've been "
            "using, takes an image, cuts it into a grid of patches, runs the patches through a "
            "vision encoder, and projects them into the very same residual stream that text flows "
            "through. From the transformer's point of view, an image is just more tokens. "
            "Which means the whole pipeline lifts over unchanged. Show the model two images, "
            "two thumbnails, two product shots, and ask which one will pull more engagement. "
            "Hook the same layer, pool the same way, build the contrastive direction from the "
            "comparisons it got right and wrong, and score new comparisons by projection. "
            "We have now benchmarked this on more than four hundred images, and the same story "
            "holds up: the contrastive direction is right there in the activations, and scoring "
            "with it beats just reading the model's raw answer. And video is the same move one more "
            "time: a video is frames, frames are images. Rank the thumbnail, rank a handful of "
            "sampled frames, and pool the scores. Text was the proving ground, but the method "
            "itself is modality agnostic."
        ),
    },
    {
        "id": "09_recap",
        "title": "The whole picture",
        "text": (
            "Let's put the whole picture on one board. Idea one: models know more than they say. "
            "So read the activations, not just the output. Idea two: to find the signal in those "
            "activations, use contrast. Average the moments the model was right, average the moments "
            "it was wrong, and subtract. That difference is a direction, and a dot product with it "
            "is a free confidence score. Idea three: once you can score, you can search. "
            "Best-of-N if you want simple; Metropolis Hastings power sampling if you want strong. "
            "That search took us from sixty-two percent to eighty-eight on LinkedIn pairs, "
            "it nudged even the brutally noisy tweet benchmark, and it carried straight over to a "
            "benchmark of four hundred plus images. No fine-tuning, no reward model, no labels at "
            "inference time. Just a model, its own hidden states, and a little bit of geometry. "
            "And I think that is the deepest takeaway here: the gap between what a model knows and "
            "what it says is real, and you can measure it, with tools as simple as a subtraction "
            "and a dot product. Thanks for watching."
        ),
    },
]
