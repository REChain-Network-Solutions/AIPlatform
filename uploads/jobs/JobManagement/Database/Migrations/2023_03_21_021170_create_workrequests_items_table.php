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
        Schema::create('workrequests_items', function (Blueprint $table) {
            $table->increments('id');
            $table->unsignedInteger('company_id')->nullable();
            $table->foreign('company_id')->references('id')->on('companies')->onDelete('cascade')->onUpdate('cascade');
            $table->unsignedInteger('workrequest_id')->nullable();
            $table->foreign('workrequest_id')->references('id')->on('workrequests')->onDelete('cascade')->onUpdate('cascade');
            $table->unsignedBigInteger('items_id')->nullable();
            $table->foreign('items_id')->references('id')->on('items')->onDelete('cascade')->onUpdate('cascade');
            $table->integer('qty')->nullable();
            $table->integer('harga')->nullable();
            $table->integer('tax')->nullable();
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
        Schema::dropIfExists('workrequests_items');
    }
};
