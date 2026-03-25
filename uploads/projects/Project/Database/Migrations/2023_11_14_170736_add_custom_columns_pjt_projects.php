<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::table('pjt_projects', function (Blueprint $table) {
            $table->string('custom_field1')->after('settings')->nullable();
            $table->string('custom_field2')->after('custom_field1')->nullable();
            $table->string('custom_field3')->after('custom_field2')->nullable();
            $table->string('custom_field4')->after('custom_field3')->nullable();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::table('', function (Blueprint $table) {

        });
    }
};
