<?php
namespace Modules\Compliance\Database\Seeders;
use Illuminate\Database\Seeder; use Illuminate\Support\Facades\DB; use Illuminate\Support\Facades\Schema;
class MenuSeeder extends Seeder {
  public function run(): void {
    $target = null;
    if (Schema::hasTable('admin_menu')) $target = 'admin_menu';
    elseif (Schema::hasTable('menus')) $target = 'menus';
    elseif (Schema::hasTable('menu_items')) $target = 'menu_items';
    if (!$target) return;
    $exists = $target === 'menu_items' ? DB::table($target)->where('url','/compliance')->exists()
                                       : DB::table($target)->where('slug','compliance')->exists();
    if (!$exists) {
      if ($target === 'menu_items') DB::table('menu_items')->insert(['title'=>'Compliance','url'=>'/compliance','icon'=>'app','sort'=>90]);
      else DB::table($target)->insert(['title'=>'Compliance','slug'=>'compliance','icon'=>'app','url'=>'/compliance','sort'=>90]);
    }
  }
}
