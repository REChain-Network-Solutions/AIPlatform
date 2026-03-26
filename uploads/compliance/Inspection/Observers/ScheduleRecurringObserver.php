<?php

namespace Modules\Inspection\Observers;

use App\Helper\Files;

use App\Models\Notification;
use Modules\Inspection\Entities\RecurringSchedule;
use Modules\Inspection\Entities\RecurringScheduleItems;


class ScheduleRecurringObserver
{

    public function saving(RecurringSchedule $schedule)
    {
        // Cannot put in creating, because saving is fired before creating. And we need company id for check bellow
        if (company()) {
            $schedule->company_id = 1;
        }
    }

    public function creating(RecurringSchedule $schedule)
    {

        if (company()) {
            $schedule->company_id = company()->id;
        }

        $days = match ($schedule->rotation) {
            'daily' => $schedule->issue_date->addDay(),
            'weekly' => $schedule->issue_date->addWeek(),
            'bi-weekly' => $schedule->issue_date->addWeeks(2),
            'monthly' => $schedule->issue_date->addMonth(),
            'quarterly' => $schedule->issue_date->addQuarter(),
            'half-yearly' => $schedule->issue_date->addMonths(6),
            'annually' => $schedule->issue_date->addYear(),
            default => $schedule->issue_date->addDay(),
        };

        $schedule->next_schedule_date = $days->format('Y-m-d');
    }

    public function created(RecurringSchedule $schedule)
    {
        if (!isRunningInConsoleOrSeeding()) {

            if (!empty(request()->item_name)) {



                foreach (request()->item_name as $key => $item) :
                    if (!is_null($item)) {
                        $recurringScheduleItem = RecurringScheduleItems::create(
                            [
                                'schedule_recurring_id' => $schedule->id,
                                'item_name' => $item

                            ]
                        );
                    }


                endforeach;
            }

        }
    }

    public function deleting(RecurringSchedule $schedule)
    {
        $notifyData = ['App\Notifications\ScheduleRecurringStatus', 'App\Notifications\NewRecurringSchedule',];
        \App\Models\Notification::deleteNotification($notifyData, $schedule->id);
    }

}
