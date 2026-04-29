<?php

namespace Database\Seeders;

use App\Models\Advertis;
use Illuminate\Database\Seeder;

class AdvertisSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        foreach (['header_left', 'header_right', 'header_bottom', 'features_bottom', 'custom_templates_bottom', 'magic_tools_bottom'] as $key) {
            Advertis::factory()->create([
                'key' => $key,
            ]);
        }
    }
}
