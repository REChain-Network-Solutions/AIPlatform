<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Support\Facades\DB;

class AddAccountingModuleVersionToSystemTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        $is_exist = DB::table('system')->where('key', 'accounting_version')->exists();

        if (! $is_exist) {
            DB::table('system')->insert([
                'key' => 'accounting_version',
                'value' => config('accounting.module_version', config('accounting.module_version')),
            ]);
        }
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        //
    }
}
