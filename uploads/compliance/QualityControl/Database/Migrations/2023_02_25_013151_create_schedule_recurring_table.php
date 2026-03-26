<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateScheduleRecurringTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        if (!Schema::hasTable('inspection_schedule_recurring')) {
            Schema::create('inspection_schedule_recurring', function (Blueprint $table) {
            $table->increments('id');
            $table->unsignedInteger('company_id')->nullable();
            $table->foreign('company_id')->references('id')->on('companies')->onDelete('set null')->onUpdate('cascade');
            $table->string('subject');
            $table->unsignedBigInteger('floor_id')->nullable();
            $table->foreign('floor_id')->references('id')->on('floors')->onDelete('set null')->onUpdate('cascade');
            $table->unsignedBigInteger('tower_id')->nullable();
            $table->foreign('tower_id')->references('id')->on('towers')->onDelete('set null')->onUpdate('cascade');
            $table->string('lokasi')->nullable();
            $table->string('shift')->nullable();
            $table->time('awal')->nullable();
            $table->time('akhir')->nullable();
            $table->mediumText('remark')->nullable();
            $table->date('issue_date');
            $table->date('next_schedule_date')->nullable();
            $table->integer('day_of_month')->nullable()->default(1);
            $table->integer('day_of_week')->nullable()->default(1);
            $table->enum('rotation', ['monthly', 'weekly', 'bi-weekly', 'quarterly', 'half-yearly', 'annually', 'daily']);
            $table->integer('billing_cycle')->nullable()->default(null);
            $table->boolean('unlimited_recurring')->default(0);
            $table->boolean('immediate_schedule')->default(false);
            $table->enum('status', ['active', 'inactive'])->default('active');
            $table->unsignedInteger('created_by');
            $table->foreign('created_by')->references('id')->on('users')->onDelete('cascade')->onUpdate('cascade');
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
        Schema::dropIfExists('inspection_schedule_recurring');
    }
}
