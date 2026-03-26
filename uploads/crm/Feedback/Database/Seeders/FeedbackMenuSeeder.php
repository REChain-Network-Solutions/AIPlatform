<?php

namespace Modules\Feedback\Database\Seeders;

use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

class FeedbackMenuSeeder extends Seeder
{
    public function run(): void
    {
        $table = config('feedback.menu_table', 'menus');
        if (!Schema::hasTable($table)) return;
        if (!DB::table($table)->where('slug','feedback')->exists()) {
            DB::table($table)->insert([
                'name' => 'Feedback & NPS',
                'slug' => 'feedback',
                'url'  => '/feedback/nps',
                'icon' => 'fa fa-star-half-o',
                'parent_id' => null,
                'order' => 51,
                'created_at' => now(),
                'updated_at' => now(),
            ]);
        }
    }
}
