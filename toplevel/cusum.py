import numpy as np


class SprtPredictor:
    '''
    An SPRT (Sequential Probability Ratio Test) novelty change point detector
    that decides as each image in a test series is encountered
    whether enough evidence of novelty has accumulated
    to declare that the change to novelty has occurred.
    For each image, it computes a score. the likelihood ratio
    of its entropy score as generated from the post-change mixed population
    over that as generated from the pre-change known population.
    Since the alpha value is unknown, the post-change alpha value
    is estimated from the scores of the seen images.
    '''

    def __init__(self, known_dist, novel_dist, scale_factor, threshold, buf_size=None, latch=True, debug=False):
        # self._known_dist, self._novel_dist = pickle.load(open(path_to_distribution, "rb"))
        self._known_dist = known_dist
        self._novel_dist = novel_dist
        self._scale_factor = scale_factor
        self._threshold = threshold
        self._buf_size = buf_size  # limit amount of history kept
        self._latch = latch
        self.reset()

    def reset(self):
        self._already_true = False
        self._count = 0
        self._known_log_probs = np.zeros(0, dtype='float')
        self._novel_log_probs = np.zeros(0, dtype='float')
        self._known_probs = np.zeros(0, dtype='float')
        self._novel_probs = np.zeros(0, dtype='float')
        self._alpha_est_sums = np.zeros(0, dtype='float')
        self._suffix_len = np.zeros(0, dtype='float')
        self._ratios = np.zeros(0, dtype='float')
        self._suffix_sums = np.zeros(0, dtype='float')

    def __call__(self, score):
        '''
        Return True if the cumulative evidence of novelty
        when you include this new image score
        exceeds (or has previously exceeded) the threshold.
        '''
        if self._known_dist == None:
            raise Exception(
                'If dists have not been set, you must call predict_from_log_probs')

        score_arr = np.array(score, dtype='float').reshape(-1, 1)
        known_log_prob = self._known_dist.score_samples(score_arr)
        novel_log_prob = self._novel_dist.score_samples(score_arr)

        return self.predict_from_log_probs(known_log_prob, novel_log_prob)

    def predict_from_log_probs(self, known_log_prob, novel_log_prob, sprt_plot_file=None):
        if self._latch and self._already_true:
            return True

        if self._buf_size:
            # Truncate structures to the last buf_size values
            self._known_log_probs = self._known_log_probs[-self._buf_size:]
            self._novel_log_probs = self._novel_log_probs[-self._buf_size:]
            self._known_probs = self._known_probs[-self._buf_size:]
            self._novel_probs = self._novel_probs[-self._buf_size:]
            self._alpha_est_sums = self._alpha_est_sums[-self._buf_size:]
            self._suffix_len = self._suffix_len[-self._buf_size:]
            self._ratios = self._ratios[-self._buf_size:]
            self._suffix_sums = self._suffix_sums[-self._buf_size:]
        
        self._known_log_probs = np.append(self._known_log_probs, known_log_prob)
        self._novel_log_probs = np.append(self._novel_log_probs, novel_log_prob)
        known_prob = np.exp(known_log_prob)
        novel_prob = np.exp(novel_log_prob)
        self._known_probs = np.append(self._known_probs, known_prob)
        self._novel_probs = np.append(self._novel_probs, novel_prob)
        alpha_est = novel_prob / (novel_prob + known_prob)
        self._alpha_est_sums = np.append(self._alpha_est_sums, 0.0)
        self._alpha_est_sums += alpha_est
        self._suffix_len = np.append(self._suffix_len, 0.0)
        self._suffix_len += 1.0
        alpha_ests = self._alpha_est_sums / self._suffix_len
        log_ratios = np.log(((1.0 - alpha_ests) * self._known_probs +
                             alpha_ests * self._novel_probs) / self._known_probs)
        
        ratio_sums = np.flip(np.cumsum(np.flip(log_ratios)))
        per_image_nov_prob = novel_prob / (novel_prob + known_prob)
        red_light_score = ratio_sums.max()
        red_light_score = max(0.0, min(1.0, (1.0 + red_light_score) / self._scale_factor))
        result = red_light_score > self._threshold
        
        if result:
            self._already_true = True
        
        return self._already_true, red_light_score, per_image_nov_prob