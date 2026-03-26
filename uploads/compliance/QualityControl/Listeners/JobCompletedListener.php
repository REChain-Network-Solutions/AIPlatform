<?php

namespace Modules\QualityControl\Listeners;

use Illuminate\Support\Arr;
use Illuminate\Support\Facades\Event;
use Modules\QualityControl\Entities\Schedule;

/**
 * Lightweight bridge for cleaning workflows.
 *
 * This listener is intentionally defensive:
 * - It does NOT assume a particular Jobs module exists.
 * - It accepts either an object event or an array payload.
 * - It only creates a QC schedule when tenant fields are present.
 */
class JobCompletedListener
{
    public function handle($event): void
    {
        $payload = $this->normalize($event);

        $companyId = Arr::get($payload, 'company_id');
        $userId    = Arr::get($payload, 'user_id');
        $jobId     = Arr::get($payload, 'job_id') ?? Arr::get($payload, 'id');

        if (!$companyId || !$userId || !$jobId) {
            // Cannot safely create tenant-scoped records.
            return;
        }

        // Idempotency: avoid creating duplicates for the same job.
        if (Schedule::where('company_id', $companyId)->where('job_id', $jobId)->exists()) {
            return;
        }

        $subject = Arr::get($payload, 'subject')
            ?? Arr::get($payload, 'title')
            ?? 'Quality Check';

        $schedule = new Schedule();
        $schedule->company_id = $companyId;
        $schedule->user_id = $userId;
        $schedule->job_id = $jobId;
        $schedule->subject = $subject;
        $schedule->issue_date = now()->toDateString();
        $schedule->status = 'open';
        $schedule->priority = 'low';
        $schedule->worker_id = Arr::get($payload, 'worker_id') ?? $userId;
        $schedule->inspect_by = Arr::get($payload, 'supervisor_id') ?? $userId;
        $schedule->remark = Arr::get($payload, 'remark');
        $schedule->save();

        // Notify downstream systems.
        Event::dispatch('quality_control.created_from_job', [
            'company_id' => $companyId,
            'user_id' => $userId,
            'job_id' => $jobId,
            'quality_control_id' => $schedule->id,
        ]);
    }

    private function normalize($event): array
    {
        if (is_array($event)) {
            return $event;
        }

        // Common patterns: $event->job, $event->model, $event->payload
        foreach (['job', 'model', 'payload'] as $prop) {
            if (is_object($event) && isset($event->{$prop})) {
                $value = $event->{$prop};
                return is_array($value) ? $value : (array) $value;
            }
        }

        return is_object($event) ? (array) $event : [];
    }
}
