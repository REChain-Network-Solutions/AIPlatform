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
        if (!Schema::hasTable('pos_payments'))
        {
            Schema::create('pos_payments', function (Blueprint $table){
                $table->bigIncrements('id');
                $table->integer('pos_id');
                $table->date('date')->nullable();
                $table->float('discount')->nullable();
                $table->string('amount')->default('0.00');
                $table->float('discount_amount')->nullable();
                $table->integer('workspace')->nullable();
                $table->integer('created_by')->default('0');
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
        Schema::dropIfExists('pos_payments');
    }
};
