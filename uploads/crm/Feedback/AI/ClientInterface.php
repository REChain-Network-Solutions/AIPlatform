<?php

namespace Modules\Feedback\AI;

interface ClientInterface
{
    public function generateFeedback(array $context): array;
    public function suggestReply(array $context): array;
    public function smoke(): array;
}
