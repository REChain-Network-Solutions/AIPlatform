<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('work_orders', function (Blueprint $table) {
            $table->id();
            $table->integer('wo_id')->default(0);
            $table->text('wo_detail')->nullable();
            $table->integer('type')->default(0);
            $table->integer('client')->default(0);
            $table->integer('asset')->default(0);
            $table->date('due_date')->nullable();
            $table->string('status')->nullable();
            $table->string('priority')->nullable();
            $table->text('notes')->nullable();
            $table->integer('assign')->default(0);
            $table->date('preferred_date')->nullable();
            $table->string('preferred_time')->nullable();
            $table->text('preferred_note')->nullable();
            $table->integer('parent_id')->default(0);
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
        Schema::dropIfExists('work_orders');
    }
};
