<?php

/*
* The MIT License
*
* Copyright (c) 2025 "YooMoney", NBСO LLC
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

namespace Tests\YooKassa\Model;

use PHPUnit\Framework\TestCase;
use YooKassa\Model\Metadata;

/**
 * MetadataTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class MetadataTest extends TestCase
{
    /**
     * @dataProvider dataProvider
     */
    public function testToArray(array $source): void
    {
        $instance = new Metadata();
        foreach ($source as $key => $value) {
            $instance->offsetSet($key, $value);
        }
        self::assertEquals($source, $instance->toArray());
    }

    /**
     * @dataProvider dataProvider
     */
    public function testCount(array $source): void
    {
        $instance = new Metadata();
        $count = 0;
        self::assertEquals($count, $instance->count());
        foreach ($source as $key => $value) {
            $instance->offsetSet($key, $value);
            ++$count;
            self::assertEquals($count, $instance->count());
        }
    }

    /**
     * @dataProvider dataProvider
     */
    public function testGetIterator(array $source): void
    {
        $instance = new Metadata();
        foreach ($source as $key => $value) {
            $instance->offsetSet($key, $value);
        }

        $iterator = $instance->getIterator();
        $tmp = $source;
        for ($iterator->rewind(); $iterator->valid(); $iterator->next()) {
            self::assertArrayHasKey($iterator->key(), $source);
            self::assertEquals($source[$iterator->key()], $iterator->current());
            unset($tmp[$iterator->key()]);
        }
        self::assertCount(0, $tmp);

        $tmp = $source;
        foreach ($instance as $key => $value) {
            self::assertArrayHasKey($key, $source);
            self::assertEquals($source[$key], $value);
            unset($tmp[$key]);
        }
        self::assertCount(0, $tmp);
    }

    public static function dataProvider()
    {
        return [
            [
                ['testKey' => 'testValue'],
            ],
            [
                [
                    'testKey1' => 'testValue1',
                    'testKey2' => 'testValue2',
                ],
            ],
            [
                [
                    'testKey1' => 'testValue1',
                    'testKey2' => 'testValue2',
                    'testKey3' => 'testValue3',
                ],
            ],
        ];
    }
}
