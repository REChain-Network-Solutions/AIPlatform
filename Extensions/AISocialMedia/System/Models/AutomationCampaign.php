<?php

namespace App\Extensions\AISocialMedia\System\Models;

use Illuminate\Database\Eloquent\Model;

class AutomationCampaign extends Model
{
    protected $table = 'automation_campaigns';

    protected $fillable = [
        'name', 'target_audience', 'user_id',
    ];

    public static function getMyCampaign()
    {
        return AutomationCampaign::where('user_id', auth()->user()->id)->get();
    }
}
