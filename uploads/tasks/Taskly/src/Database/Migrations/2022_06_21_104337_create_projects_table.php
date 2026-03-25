<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateProjectsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        if(!Schema::hasTable('projects'))
        {
            Schema::create('projects', function (Blueprint $table) {
                $table->bigIncrements('id');
                $table->string('name');
                $table->enum('status',['Ongoing','Finished','OnHold'])->default('Ongoing');
                $table->string('image')->nullable();
                $table->longText('description')->nullable();
                $table->date('start_date')->nullable();
                $table->date('end_date')->nullable();
                $table->integer('budget')->default(0);
                $table->integer('is_active')->default(1);
                $table->string('type')->default('project');
                $table->string('currency', 50)->default('$');
                $table->string('project_progress', 6)->default('false');
                $table->string('progress', 5)->default(0);
                $table->string('task_progress', 6)->default('true');
                $table->text('tags')->nullable();
                $table->integer('estimated_hrs')->default(0);
                $table->longText('copylinksetting')->nullable();
                $table->string('password')->nullable();
                $table->integer('workspace');
                $table->integer('created_by');
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
        Schema::dropIfExists('projects');
    }
}
