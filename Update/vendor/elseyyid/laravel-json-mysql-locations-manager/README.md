# Laravel Json Localization Manager

This is a package to manage json strings for Laravel 5.4 and higher Localization Translation Strings As Keys With MySQL.
[Using Translation Strings As Keys][6ea2d96e]

  [6ea2d96e]: https://laravel.com/docs/5.4/localization#using-translation-strings-as-keys "Using Translation Strings As Keys"

## Composer require
``` bash
$ composer require elseyyid/laravel-json-mysql-locations-manager
```
## Add Provider into config/app.php
``` php
Elseyyid\LaravelJsonLocationsManager\Providers\LaravelJsonLocationsManagerServiceProvider::class,
```
### For Only local environment
Add the following to the AppServiceProvider in the register function:
#### app/Providers/AppServiceProvider.php
``` php
 if ($this->app->environment() == 'local' || $this->app->environment() == 'testing') {
     /*
      * Load third party local providers
      */
     $this->app->register(\Elseyyid\LaravelJsonLocationsManager\Providers\LaravelJsonLocationsManagerServiceProvider::class);
 }
```

## Package install
Run in console this command
``` bash
$ php artisan elseyyid:location:install
```
This command install database and will ask if you want to import the existing locations in the application.
There are 2 questions, the first to import the strings in arrays and the second to import the existing strings into json files.
This will pupulate the database with all strings. You don't need to run artisan migrate, this package uses an independent database (sqlite).


### Other Console Command
You can publish all languages Json files in console running
``` bash
$ php artisan elseyyid:location:publish
```
Of course you also can do that in browser views.

## Manage Locations
Now you can access to /translations/home in your browser and manage all langs strings.
 - Search Language Strings
 - Add new Language
 - Add new Strings
 - Edit Strings
 - Publish / Update json Files

## Publish the views and config file
``` bash
$ php artisan vendor:publish --provider='Elseyyid\LaravelJsonLocationsManager\Providers\LaravelJsonLocationsManagerServiceProvider'
```
### In the config file you can call your custom layout, the content section and the scripts section (this is important for the edit views) and routes prefix and middlewares.

## Routes middlewares
in config file you can add your middlewares, by default is only 'web' middleware

### Feel free to send improvements
Created by [elseyyid][760a7857]

  [760a7857]: https://github.com/OmarElseyyid "https://github.com/OmarElseyyid"


