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
        Schema::create('workorders', function (Blueprint $table) {
            $table->increments('id');
            $table->unsignedInteger('company_id')->nullable();
            $table->foreign('company_id')->references('id')->on('companies')->onDelete('set null')->onUpdate('cascade');
            $table->unsignedInteger('workrequest_id')->nullable();
            $table->foreign('workrequest_id')->references('id')->on('workrequests')->onDelete('set null')->onUpdate('cascade');
            $table->unsignedInteger('complaint_id')->nullable();
            $table->foreign('complaint_id')->references('id')->on('complaint')->onDelete('cascade')->onUpdate('cascade');
            $table->unsignedInteger('invoice_id')->nullable();
            $table->foreign('invoice_id')->references('id')->on('invoices')->onDelete('cascade')->onUpdate('cascade');
            $table->unsignedInteger('workorder_recurring_id')->nullable();
            $table->foreign('workorder_recurring_id')->references('id')->on('workorders_recurring')->onDelete('cascade')->onUpdate('cascade');
            $table->string('nomor_wo'); 
            $table->enum('category', ['planned', 'unplanned'])->default('unplanned');
            $table->enum('priority', ['low', 'medium', 'high', 'emergency'])->default('medium');
            $table->enum('status', ['incomplete', 'pending','completed'])->default('incomplete');
            $table->mediumText('work_description')->nullable();
            $table->dateTime('schedule_start')->nullable();
            $table->dateTime('schedule_finish')->nullable();            
            $table->integer('estimate_hours')->nullable();
            $table->integer('estimate_minutes')->nullable();            
            $table->dateTime('actual_start')->nullable();
            $table->dateTime('actual_finish')->nullable();
            $table->integer('actual_hours')->nullable();
            $table->integer('actual_minutes')->nullable();
            $table->mediumText('completion_notes')->nullable();
            $table->boolean('tenant_approval')->default(0);
            $table->unsignedInteger('house_id')->nullable();
            $table->foreign('house_id')->references('id')->on('houses')->onDelete('cascade')->onUpdate('cascade'); 
            $table->unsignedInteger('assets_id')->nullable();
            $table->foreign('assets_id')->references('id')->on('assets')->onDelete('cascade')->onUpdate('cascade'); 
            $table->mediumText('problem')->nullable();
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
        Schema::dropIfExists('workorders');
    }
};
