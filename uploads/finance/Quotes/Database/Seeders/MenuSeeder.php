<?php

namespace Modules\Quotes\Database\Seeders;

use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;

class MenuSeeder extends Seeder
{
    public function run(): void
    {
        $now = now();
        $data = [
            'title' => 'Quotes & Estimates',
            'url'   => '/quotes',
            'icon'  => 'fa fa-file-lines',
            'order' => 60,
            'updated_at' => $now,
        ];
        $exists = DB::table('menus')->where('slug', 'quotes')->exists();
        if ($exists) {
            DB::table('menus')->where('slug', 'quotes')->update($data);
        } else {
            DB::table('menus')->insert(array_merge(['slug' => 'quotes', 'created_at' => $now], $data));
        }
    }
}
