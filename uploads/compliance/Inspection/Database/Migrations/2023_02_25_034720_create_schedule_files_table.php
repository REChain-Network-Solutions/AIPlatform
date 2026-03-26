<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateScheduleFilesTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        if (!Schema::hasTable('inspection_schedule_files')) {
            Schema::create('inspection_schedule_files', function (Blueprint $table) {
            $table->increments('id');
            $table->unsignedInteger('user_id')->nullable();
            $table->foreign('user_id')->references('id')->on('users')->onDelete('cascade')->onUpdate('cascade');
            $table->unsignedInteger('schedule_reply_id')->nullable();
            $table->foreign('schedule_reply_id')->references('id')->on('inspection_schedule_replies')->onDelete('cascade')->onUpdate('cascade');
            $table->string('filename');
            $table->text('description')->nullable()->default(null);
            $table->string('google_url')->nullable()->default(null);
            $table->string('hashname')->nullable()->default(null);
            $table->string('size')->nullable()->default(null);
            $table->string('dropbox_link')->nullable()->default(null);
            $table->string('external_link')->nullable()->default(null);
            $table->string('external_link_name')->nullable()->default(null);
            $table->timestamps();
        });
        }
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('inspection_schedule_files');
    }
}
