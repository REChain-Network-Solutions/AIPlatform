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
        Schema::create('workorders_recurring', function (Blueprint $table) {
            $table->increments('id');
            $table->unsignedInteger('company_id')->nullable();
            $table->foreign('company_id')->references('id')->on('companies')->onDelete('set null')->onUpdate('cascade');
            $table->unsignedInteger('workrequest_id')->nullable();
            $table->foreign('workrequest_id')->references('id')->on('workrequests')->onDelete('set null')->onUpdate('cascade');
            $table->unsignedInteger('ticket_id')->nullable();
            $table->foreign('ticket_id')->references('id')->on('tickets')->onDelete('cascade')->onUpdate('cascade');
            $table->unsignedInteger('invoice_id')->nullable();
            $table->foreign('invoice_id')->references('id')->on('invoices')->onDelete('cascade')->onUpdate('cascade');
            $table->enum('category', ['planned', 'unplanned'])->default('unplanned');
            $table->enum('priority', ['low', 'medium', 'high', 'emergency'])->default('medium');
            $table->enum('status_wo', ['incomplete', 'pending','completed'])->default('incomplete');
            $table->mediumText('work_description')->nullable();
            $table->dateTime('schedule_start')->nullable();
            $table->dateTime('schedule_finish')->nullable();            
            $table->integer('estimate_hours')->nullable();
            $table->integer('estimate_minutes')->nullable();            
            $table->date('issue_date');
            $table->date('next_schedule_date')->nullable();
            $table->integer('day_of_month')->nullable()->default(1);
            $table->integer('day_of_week')->nullable()->default(1);
            $table->enum('rotation', ['monthly', 'weekly', 'bi-weekly', 'quarterly', 'half-yearly', 'annually', 'daily']);
            $table->integer('billing_cycle')->nullable()->default(null);
            $table->boolean('unlimited_recurring')->default(0);
            $table->boolean('immediate_schedule')->default(false);
            $table->enum('status', ['active', 'inactive'])->default('active');
            $table->unsignedInteger('unit_id')->nullable();
            $table->foreign('unit_id')->references('id')->on('units')->onDelete('cascade')->onUpdate('cascade'); 
            $table->unsignedInteger('assets_id')->nullable();
            $table->foreign('assets_id')->references('id')->on('assets')->onDelete('cascade')->onUpdate('cascade'); 
            $table->unsignedInteger('created_by');
            $table->foreign('created_by')->references('id')->on('users')->onDelete('cascade')->onUpdate('cascade');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('workorders_recurring');
    }
};
