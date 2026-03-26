<?php

namespace Modules\Complaint\Observers;

use Modules\Complaint\Events\ComplaintEvent;
use Modules\Complaint\Events\ComplaintRequesterEvent;
use App\Models\Notification;
use App\Models\Order;
use Modules\Complaint\Entities\Complaint;
use App\Models\UniversalSearch;

class ComplaintObserver
{
    public function saving(Complaint $complaint)
    {
        if (!isRunningInConsoleOrSeeding()) {
            $userID = (!is_null(user())) ? user()->id : $complaint->user_id;
            $complaint->last_update_by = $userID;
        }
    }

    public function creating(Complaint $model)
    {
        if (!isRunningInConsoleOrSeeding()) {
            $userID = (!is_null(user())) ? user()->id : $model->user_id;
            $model->added_by = $userID;

            if ($model->isDirty('status') && $model->status == 'closed') {
                $model->close_date = now(company()->timezone)->format('Y-m-d');
            }

        }

        if (company()) {
            $model->company_id = company()->id;
        }

        $model->complaint_number = (int)Complaint::max('complaint_number') + 1;

    }

    public function created(Complaint $model)
    {
        if (!isRunningInConsoleOrSeeding()) {
            // Send admin notification
            event(new ComplaintEvent($model, 'NewComplaint'));

            if ($model->requester) {
                event(new ComplaintRequesterEvent($model, $model->requester));
            }

            if ($model->agent_id != '') {
                event(new ComplaintEvent($model, 'ComplaintAgent'));
            }

        }
    }

    public function updating(Complaint $complaint)
    {
        if (!isRunningInConsoleOrSeeding()) {
            if ($complaint->isDirty('status') && $complaint->status == 'closed') {
                $complaint->close_date = now(company()->timezone)->format('Y-m-d');
            }
        }
    }

    public function updated(Complaint $complaint)
    {
        if (!isRunningInConsoleOrSeeding()) {
            if ($complaint->isDirty('agent_id') && $complaint->agent_id != '') {
                event(new ComplaintEvent($complaint, 'ComplaintAgent'));
            }
        }
    }

    public function deleting(Complaint $complaint)
    {
        $universalSearches = UniversalSearch::where('searchable_id', $complaint->id)->where('module_type', 'complaint')->get();

        if ($universalSearches) {
            foreach ($universalSearches as $universalSearch) {
                UniversalSearch::destroy($universalSearch->id);
            }
        }

        $notifyData = ['Modules\Complaint\Notifications\NewComplaint', 'Modules\Complaint\Notifications\NewComplaintReply', 'Modules\Complaint\Notifications\NewComplaintRequester', 'Modules\Complaint\Notifications\ComplaintAgent'];

        \App\Models\Notification::deleteNotification($notifyData, $complaint->id);

    }

}
