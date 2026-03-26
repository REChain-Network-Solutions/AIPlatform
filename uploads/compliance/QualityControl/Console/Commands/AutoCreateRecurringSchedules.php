<?php

namespace Modules\QualityControl\Console\Commands;

use App\Models\Company;
use Carbon\Carbon;
use App\Models\UniversalSearch;
use App\Scopes\CompanyScope;
use Illuminate\Console\Command;
use App\Scopes\ActiveScope;
use App\Traits\UniversalSearchTrait;
use Illuminate\Support\Facades\DB;
use Modules\QualityControl\Entities\ScheduleItems;
use Modules\QualityControl\Entities\RecurringSchedule;
use Modules\QualityControl\Entities\Schedule;
use App\Models\Event;

class AutoCreateRecurringSchedules extends Command
{

    use UniversalSearchTrait;

    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'recurring-schedule-create';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'auto create recurring inspection_schedules ';

    /**
     * Execute the console command.
     *
     * @return mixed
     */

    public function handle()
    {
        $recurringSchedules = RecurringSchedule::with(['recurrings'])->where('status', 'active')->get();

        foreach ($recurringSchedules as $recurring) {

            if (is_null($recurring->next_schedule_date)) {
                continue;
            }

            $totalExistingCount = $recurring->recurrings->count();

            if ($recurring->unlimited_recurring == 1 || ($totalExistingCount < $recurring->billing_cycle)) {

                if ($recurring->next_schedule_date->timezone($recurring->company->timezone)->isToday()) {
                    $this->makeSchedule($recurring);
                    $this->saveNextScheduleDate($recurring);
                }
            }
        }
    }

    private function saveNextScheduleDate($recurring)
    {
        $days = match ($recurring->rotation) {
            'daily' => now()->addDay(),
            'weekly' => now()->addWeek(),
            'bi-weekly' => now()->addWeeks(2),
            'monthly' => now()->addMonth(),
            'quarterly' => now()->addQuarter(),
            'half-yearly' => now()->addMonths(6),
            'annually' => now()->addYear(),
            default => now()->addDay(),
        };

        $recurring->next_schedule_date = $days->format('Y-m-d');
        $recurring->save();
    }

    /**
     * Execute the console command.
     *
     * @return mixed
     */

    public function makeSchedule($scheduleData)
    {

        $recurring = $scheduleData;

        $schedule = new Schedule();
        $schedule->schedule_recurring_id = $recurring->id;
        $schedule->company_id = $recurring->company_id;
        $schedule->issue_date = now()->format('Y-m-d');
        $schedule->subject = $recurring->subject;
        $schedule->floor_id = $recurring->floor_id;
        $schedule->tower_id = $recurring->tower_id;
        $schedule->lokasi = $recurring->lokasi;
        $schedule->shift = $recurring->shift;
        $schedule->awal = $recurring->awal;
        $schedule->akhir = $recurring->akhir;

        $schedule->save();


        // Sync to WorkSuite calendar
        if (class_exists(\App\Models\Event::class)) {
            $date = $schedule->issue_date ?? now()->format('Y-m-d');
            $startTime = $schedule->awal ? Carbon::parse($schedule->awal)->format('H:i:s') : '09:00:00';
            $endTime = $schedule->akhir ? Carbon::parse($schedule->akhir)->format('H:i:s') : null;

            $start = Carbon::parse($date . ' ' . $startTime);
            $end = $endTime ? Carbon::parse($date . ' ' . $endTime) : (clone $start)->addHour();
            if ($end->lessThanOrEqualTo($start)) {
                $end = (clone $start)->addHour();
            }

            $event = new Event();
            $event->company_id = $schedule->company_id;
            $event->scheduled_service_id = $schedule->id;
            $event->event_type = 'inspection';
            $event->service_status = 'scheduled';
            $event->event_name = $schedule->subject ?? 'Inspection';
            $event->label_color = '#F57F25';
            $event->where = $schedule->lokasi ?? '';
            $event->description = '';
            $event->note = ($schedule->shift ? ('Shift: ' . $schedule->shift) : '');
            $event->start_date_time = $start->format('Y-m-d H:i:s');
            $event->end_date_time = $end->format('Y-m-d H:i:s');
            $event->save();
        }



        foreach ($recurring->items as $item) {

            $scheduleItem = ScheduleItems::create(
                [
                    'schedule_id' => $schedule->id,
                    'item_name' => $item->item_name
                ]
            );
        }

        // Log search
        $this->logSearchEntry($schedule->id, $schedule->subject, 'inspection_schedules.show', 'schedule');
    }

}
