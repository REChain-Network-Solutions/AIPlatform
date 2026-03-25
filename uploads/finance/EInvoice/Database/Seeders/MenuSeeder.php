<?php

namespace Modules\EInvoice\Database\Seeders;

use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;

class MenuSeeder extends Seeder
{
    public function run(): void
    {
        $now = now();

        // --- Main E-Invoices link ---
        $main = [
            'title' => 'E-Invoices',
            'url'   => '/einvoice',
            'icon'  => 'fa fa-file-invoice',
            'order' => 50,
            'updated_at' => $now,
        ];
        $exists = DB::table('menus')->where('slug', 'einvoice')->exists();
        if ($exists) {
            DB::table('menus')->where('slug', 'einvoice')->update($main);
        } else {
            DB::table('menus')->insert(array_merge(['slug' => 'einvoice', 'created_at' => $now], $main));
        }

        // --- E-Invoice AI settings link ---
        $ai = [
            'title' => 'E-Invoice AI',
            'url'   => '/einvoice/settings/ai',
            'icon'  => 'fa fa-robot',
            'order' => 51,
            'updated_at' => $now,
        ];
        $existsAi = DB::table('menus')->where('slug', 'einvoice-ai')->exists();
        if ($existsAi) {
            DB::table('menus')->where('slug', 'einvoice-ai')->update($ai);
        } else {
            DB::table('menus')->insert(array_merge(['slug' => 'einvoice-ai', 'created_at' => $now], $ai));
        }
    }
}
