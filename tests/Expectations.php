<?php

declare(strict_types=1);
use Pest\Expectation;

expect()->extend('seeHtml', function ($values): Expectation|ExpectationsInterface {
    test()->assertSeeHtml($this->value, $values);

    return $this;
});

expect()->extend('dontSeeHtml', function ($values): Expectation|ExpectationsInterface {
    test()->assertDontSeeHtml($this->value, $values);

    return $this;
});

expect()->extend('seeHtmlInOrder', function (array $values): Expectation|ExpectationsInterface {
    test()->assertSeeHtmlInOrder($this->value, $values);

    return $this;
});
