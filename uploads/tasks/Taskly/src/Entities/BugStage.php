<?php

namespace Workdo\Taskly\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class BugStage extends Model
{
    use HasFactory;

    protected $fillable = [
        'id',
        'name',
         'color',
         'complete',
         'workspace_id',
         'order',
         'created_by'
    ];

    protected static function newFactory()
    {
        return \Workdo\Taskly\Database\factories\BugStageFactory::new();
    }
    static function defultadd(){
        $defaultStages = [
            '#77b6ea' => __('Unconfirmed'),
            '#6e00ff' => __('Confirmed'),
            '#3cb8d9' => __('In Progress'),
            '#37b37e' => __('Resolved'),
            '#545454' => __('Verified'),
        ];
        $key = 0;
        $lastKey       = count($defaultStages) - 1;
        foreach($defaultStages as $color => $stage)
        {
            BugStage::create([
                'name' => $stage,
                'color' => $color,
                'workspace_id' => getActiveWorkSpace(),
                'created_by' => creatorId(),
                'complete' => ($key == $lastKey) ? true : false,
                'order' => $key,
            ]);
            $key++;
        }
    }
}
