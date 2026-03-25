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
        if (!Schema::hasTable('pos_products'))
        {
            Schema::create('pos_products', function (Blueprint $table) {
                $table->bigIncrements('id');
                $table->integer('pos_id')->default('0');
                $table->integer('product_id')->default('0');
                $table->integer('quantity')->default('0');
                $table->string('tax')->default('0');
                $table->float('discount')->default('0.00');
                $table->float('price')->default('0.00');
                $table->text('description')->nullable();
                $table->integer('workspace')->nullable();
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
        Schema::dropIfExists('pos_products');
    }
};
