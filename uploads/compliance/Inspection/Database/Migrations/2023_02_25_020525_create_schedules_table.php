<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateSchedulesTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        if (!Schema::hasTable('inspection_schedules')) {
            Schema::create('inspection_schedules', function (Blueprint $table) {
            $table->increments('id');
            $table->unsignedInteger('company_id')->nullable();
            $table->foreign('company_id')->references('id')->on('companies')->onDelete('set null')->onUpdate('cascade');
            $table->unsignedInteger('schedule_recurring_id')->nullable();
            $table->foreign('schedule_recurring_id')->references('id')->on('inspection_schedule_recurring')->onDelete('set null')->onUpdate('cascade');
            $table->string('subject')->nullable();
            $table->integer('floor_id')->unsigned()->nullable();
            $table->foreign('floor_id')->references('id')->on('floors')->onDelete('set null')->onUpdate('cascade');
            $table->integer('tower_id')->unsigned()->nullable();
            $table->foreign('tower_id')->references('id')->on('towers')->onDelete('set null')->onUpdate('cascade');
            $table->string('lokasi')->nullable();
            $table->string('shift')->nullable();
            $table->time('awal')->nullable();
            $table->time('akhir')->nullable();
            $table->date('issue_date')->nullable();
            $table->datetime('inspected_at')->nullable();
            $table->datetime('closed_at')->nullable();
            $table->string('pic')->nullable();
            $table->mediumText('remark')->nullable();
            $table->enum('status', ['open', 'pending', 'resolved', 'closed'])->default('open');
            $table->enum('priority', ['low', 'medium', 'high', 'urgent'])->default('low');
            $table->unsignedInteger('worker_id')->nullable();
            $table->foreign('worker_id')->references('id')->on('users')->onDelete('set null')->onUpdate('cascade');
            $table->unsignedInteger('inspect_by')->nullable();
            $table->foreign('inspect_by')->references('id')->on('users')->onDelete('set null')->onUpdate('cascade');
            $table->unsignedInteger('agent_id')->nullable();
            $table->foreign('agent_id')->references('id')->on('users')->onDelete('set null')->onUpdate('cascade');
            $table->timestamps();
            $table->softDeletes();
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
        Schema::dropIfExists('inspection_schedules');
    }
}
