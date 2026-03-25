<?php

namespace Workdo\Taskly\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class Stage extends Model
{
    use HasFactory;

    protected $fillable = [
        'name', 'color', 'complete', 'workspace_id', 'order','created_by',
    ];

    protected static function newFactory()
    {
        return \Workdo\Taskly\Database\factories\StageFactory::new();
    }
    static function defultadd(){
        $defaultStages = [
            '#77b6ea' => __('Todo'),
            '#545454' => __('In Progress'),
            '#3cb8d9' => __('Review'),
            '#37b37e' => __('Done'),
        ];
        $key = 0;
        $lastKey       = count($defaultStages) - 1;
        foreach($defaultStages as $color => $stage)
        {
            Stage::create([
                    'name' => $stage,
                    'color' => $color,
                    'workspace_id' => getActiveWorkSpace(),
                    'complete' => ($key == $lastKey) ? true : false,
                    'order' => $key,
                ]);
            $key++;
        }
    }
}
