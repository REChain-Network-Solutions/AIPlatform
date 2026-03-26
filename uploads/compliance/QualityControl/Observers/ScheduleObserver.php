<?php

namespace Modules\QualityControl\Observers;


use Exception;
use App\Models\User;
use App\Helper\Files;
use App\Scopes\ActiveScope;
use Illuminate\Support\Str;
use App\Models\Notification;

use App\Models\CompanyAddress;

use App\Models\UniversalSearch;
use Modules\QualityControl\Entities\Schedule;
use Modules\QualityControl\Entities\ScheduleItems;

class ScheduleObserver
{

    public function saving(Schedule $schedule)
    {

    }

    public function creating(Schedule $schedule)
    {

        if (company()) {
            $schedule->company_id = company()->id;
        }
    }

    public function created(Schedule $schedule)
    {
        // if (!isRunningInConsoleOrSeeding()) {
        //     if (!empty(request()->item_name) && is_array(request()->item_name)) {

        //         foreach (request()->item_name as $key => $item) :
        //             if (!is_null($item)) {
        //                 $scheduleItem = ScheduleItems::create(
        //                     [
        //                         'schedule_id' => $schedule->id,
        //                         'item_name' => $item
        //                     ]
        //                 );
        //             }



        //         endforeach;
        //     }





        // }

        // $schedule->saveQuietly();

    }

    public function updating(Schedule $schedule)
    {

    }

    public function updated(Schedule $schedule)
    {
        // if (!isRunningInConsoleOrSeeding()) {
        //     /*
        //         Step1 - Delete all schedule items which are not avaialable
        //         Step2 - Find old inspection_schedules items, update it and check if images are newer or older
        //         Step3 - Insert new inspection_schedules items with images
        //     */

        //     $request = request();

        //     $items = $request->item_name;
        //     $item_ids = $request->item_ids;

        //     if (!empty($request->item_name) && is_array($request->item_name) ) {
        //         // Step1 - Delete all schedule items which are not avaialable
        //         if (!empty($item_ids)) {
        //             ScheduleItems::whereNotIn('id', $item_ids)->where('schedule_id', $schedule->id)->delete();
        //         }

        //         // Step2&3 - Find old inspection_schedules items, update it and check if images are newer or older
        //         foreach ($items as $key => $item) {
        //             $schedule_item_id = isset($item_ids[$key]) ? $item_ids[$key] : 0;

        //             try {
        //                 $scheduleItem = ScheduleItems::findOrFail($schedule_item_id);
        //             }
        //             catch(Exception )  {
        //                     $scheduleItem = new ScheduleItems();
        //             }

        //             $scheduleItem->schedule_id = $schedule->id;
        //             $scheduleItem->item_name = $item;

        //             $scheduleItem->saveQuietly();


        //         }
        //     }
        // }


        // $schedule->saveQuietly();



    }

    public function deleting(Schedule $schedule)
    {
        $universalSearches = UniversalSearch::where('searchable_id', $schedule->id)->where('module_type', 'schedule')->get();

        if ($universalSearches) {
            foreach ($universalSearches as $universalSearch) {
                UniversalSearch::destroy($universalSearch->id);
            }
        }

    }



}
