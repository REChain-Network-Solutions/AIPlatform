<?php

namespace App\Extensions\AISocialMedia\System\Models;

use Illuminate\Database\Eloquent\Model;

class Automation extends Model
{
    protected $table = 'automations';

    protected $fillable = [
        'custom',
        'value',
    ];

    public static function getActivePlatforms()
    {
        if (Automation::where('custom', 'active_platforms')->exists()) {
            $active_platforms = Automation::where('custom', 'active_platforms')->first()->value;

            return json_decode($active_platforms);
        }

        $active_platforms = [
            [
                'id'   => '1',
                'name' => 'X',
                'logo' => 'images/platforms/x-logo.png',
            ],
            [
                'id'   => '2',
                'name' => 'LinkedIn',
                'logo' => 'images/platforms/linkedin-logo.png',
            ],
        ];
        Automation::create([
            'custom' => 'active_platforms',
            'value'  => json_encode($active_platforms),
        ]);

        return json_decode(json_encode($active_platforms));
    }
}
