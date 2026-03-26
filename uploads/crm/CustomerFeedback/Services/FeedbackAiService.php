<?php

namespace Modules\CustomerFeedback\Services;

use Modules\CustomerFeedback\Entities\FeedbackInsight;
use Modules\CustomerFeedback\Entities\FeedbackTicket;

class FeedbackAiService
{
    public function analyzeTicket(FeedbackTicket $ticket): array
    {
        $sentiment = $this->sentimentLabel($ticket);
        $priority = $this->prioritySuggestion($ticket);
        $category = $this->categorySuggestion($ticket);
        $response = $this->suggestResponse($ticket);

        $insights = [
            [
                'insight_type' => FeedbackInsight::TYPE_SENTIMENT,
                'title' => 'Sentiment analysis',
                'description' => 'Detected sentiment: ' . $sentiment['label'],
                'confidence_score' => $sentiment['confidence'],
                'suggested_action' => $sentiment['label'] === 'negative' ? 'Prioritise rapid follow-up.' : 'Continue normal handling.',
                'tags' => [$sentiment['label']],
                'metadata' => $sentiment,
            ],
            [
                'insight_type' => FeedbackInsight::TYPE_CATEGORY,
                'title' => 'Suggested category',
                'description' => 'Suggested category: ' . $category,
                'confidence_score' => 0.72,
                'suggested_action' => 'Review and confirm routing.',
                'tags' => [$category],
                'metadata' => ['category' => $category],
            ],
            [
                'insight_type' => FeedbackInsight::TYPE_PRIORITY,
                'title' => 'Suggested priority',
                'description' => 'Suggested priority: ' . $priority,
                'confidence_score' => 0.78,
                'suggested_action' => 'Adjust assignee SLA if required.',
                'tags' => [$priority],
                'metadata' => ['priority' => $priority],
            ],
            [
                'insight_type' => FeedbackInsight::TYPE_ACTION,
                'title' => 'Suggested response',
                'description' => $response,
                'confidence_score' => 0.69,
                'suggested_action' => $response,
                'tags' => ['response'],
                'metadata' => ['response' => $response],
            ],
        ];

        foreach ($insights as $payload) {
            FeedbackInsight::updateOrCreate(
                [
                    'feedback_ticket_id' => $ticket->id,
                    'insight_type' => $payload['insight_type'],
                    'company_id' => $ticket->company_id,
                ],
                $payload
            );
        }

        return [
            'sentiment' => $sentiment,
            'priority' => $priority,
            'category' => $category,
            'response' => $response,
        ];
    }

    public function getInsights(FeedbackTicket $ticket)
    {
        return $ticket->insights()->latest()->get();
    }

    public function getSentiment(FeedbackTicket $ticket): array
    {
        return $this->sentimentLabel($ticket);
    }

    public function suggestCategory(FeedbackTicket $ticket): string
    {
        return $this->categorySuggestion($ticket);
    }

    public function suggestPriority(FeedbackTicket $ticket): string
    {
        return $this->prioritySuggestion($ticket);
    }

    public function suggestResponse(FeedbackTicket $ticket): string
    {
        $base = 'Thanks for your feedback. We have logged your request and assigned it for review.';
        if ($this->sentimentLabel($ticket)['label'] === 'negative') {
            $base = 'We are sorry about your experience. Your case has been escalated and a team member will contact you shortly.';
        }
        if ($ticket->isSurveyResponse()) {
            $base = 'Thank you for completing the survey. We appreciate the detail and will use it to improve service quality.';
        }

        return $base;
    }

    private function sentimentLabel(FeedbackTicket $ticket): array
    {
        $text = strtolower(($ticket->title ?? '') . ' ' . ($ticket->description ?? ''));
        $negativeWords = ['bad', 'poor', 'late', 'angry', 'broken', 'terrible', 'issue', 'problem', 'complaint', 'refund'];
        $positiveWords = ['great', 'good', 'happy', 'excellent', 'love', 'amazing', 'thanks'];

        $negative = 0;
        $positive = 0;

        foreach ($negativeWords as $word) {
            $negative += substr_count($text, $word);
        }
        foreach ($positiveWords as $word) {
            $positive += substr_count($text, $word);
        }

        if ($negative > $positive) {
            return ['label' => 'negative', 'confidence' => 0.81];
        }
        if ($positive > $negative) {
            return ['label' => 'positive', 'confidence' => 0.76];
        }
        return ['label' => 'neutral', 'confidence' => 0.64];
    }

    private function categorySuggestion(FeedbackTicket $ticket): string
    {
        $text = strtolower(($ticket->title ?? '') . ' ' . ($ticket->description ?? ''));
        if (str_contains($text, 'refund') || str_contains($text, 'invoice') || str_contains($text, 'bill')) {
            return 'billing';
        }
        if (str_contains($text, 'late') || str_contains($text, 'schedule') || str_contains($text, 'missed')) {
            return 'service-delivery';
        }
        if (str_contains($text, 'staff') || str_contains($text, 'agent') || str_contains($text, 'team')) {
            return 'team-quality';
        }
        return $ticket->feedback_type === FeedbackTicket::TYPE_COMPLAINT ? 'complaint' : 'general-feedback';
    }

    private function prioritySuggestion(FeedbackTicket $ticket): string
    {
        $text = strtolower(($ticket->title ?? '') . ' ' . ($ticket->description ?? ''));
        if (str_contains($text, 'urgent') || str_contains($text, 'asap') || str_contains($text, 'immediately')) {
            return FeedbackTicket::PRIORITY_CRITICAL;
        }
        if (str_contains($text, 'refund') || str_contains($text, 'broken') || str_contains($text, 'unsafe')) {
            return FeedbackTicket::PRIORITY_HIGH;
        }
        return $ticket->priority ?: FeedbackTicket::PRIORITY_MEDIUM;
    }
}
