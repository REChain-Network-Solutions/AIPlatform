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

namespace Tests\YooKassa\Request\Deals;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Model\Metadata;
use YooKassa\Request\Deals\DealsRequest;

/**
 * DealsRequestTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class DealsRequestTest extends AbstractTestCase
{
    protected DealsRequest $object;

    /**
     * @param mixed|null $value
     * @return DealsRequest
     */
    protected function getTestInstance(mixed $value = null): DealsRequest
    {
        return new DealsRequest($value);
    }

    /**
     * @return void
     */
    public function testDealsRequestClassExists(): void
    {
        $this->object = $this->getMockBuilder(DealsRequest::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(DealsRequest::class));
        $this->assertInstanceOf(DealsRequest::class, $this->object);
        self::assertTrue($this->object->validate());
    }

    /**
     * Test property "cursor"
     * @dataProvider validCursorDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCursor(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCursor());
        self::assertEmpty($instance->cursor);
        $instance->setCursor($value);
        self::assertEquals($value, is_array($value) ? $instance->getCursor()->toArray() : $instance->getCursor());
        self::assertEquals($value, is_array($value) ? $instance->cursor->toArray() : $instance->cursor);
        if (!empty($value)) {
            self::assertTrue($instance->hasCursor());
            self::assertNotNull($instance->getCursor());
            self::assertNotNull($instance->cursor);
        }
    }

    /**
     * Test invalid property "cursor"
     * @dataProvider invalidCursorDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCursor(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCursor($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCursorDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_cursor'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCursorDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_cursor'));
    }

    /**
     * Test property "limit"
     * @dataProvider validLimitDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testLimit(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setLimit($value);
        self::assertEquals($value, is_array($value) ? $instance->getLimit()->toArray() : $instance->getLimit());
        self::assertEquals($value, is_array($value) ? $instance->limit->toArray() : $instance->limit);
        if (!empty($value)) {
            self::assertTrue($instance->hasLimit());
            self::assertNotNull($instance->getLimit());
            self::assertNotNull($instance->limit);
            self::assertLessThanOrEqual(100, is_string($instance->getLimit()) ? mb_strlen($instance->getLimit()) : $instance->getLimit());
            self::assertLessThanOrEqual(100, is_string($instance->limit) ? mb_strlen($instance->limit) : $instance->limit);
            self::assertGreaterThanOrEqual(1, is_string($instance->getLimit()) ? mb_strlen($instance->getLimit()) : $instance->getLimit());
            self::assertGreaterThanOrEqual(1, is_string($instance->limit) ? mb_strlen($instance->limit) : $instance->limit);
            self::assertIsNumeric($instance->getLimit());
            self::assertIsNumeric($instance->limit);
        }
    }

    /**
     * Test invalid property "limit"
     * @dataProvider invalidCursorDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidLimit(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCursor($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validLimitDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_limit'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidLimitDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_limit'));
    }

    /**
     * Test property "created_at_gte"
     * @dataProvider validCreatedAtGteDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCreatedAtGte(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCreatedAtGte());
        self::assertEmpty($instance->created_at_gte);
        $instance->setCreatedAtGte($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCreatedAtGte());
            self::assertNotNull($instance->getCreatedAtGte());
            self::assertNotNull($instance->created_at_gte);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCreatedAtGte());
                self::assertEquals($value, $instance->created_at_gte);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCreatedAtGte());
                self::assertEquals(new Datetime($value), $instance->created_at_gte);
            }
        }
    }

    /**
     * Test invalid property "created_at_gte"
     * @dataProvider invalidCreatedAtGteDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCreatedAtGte(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCreatedAtGte($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCreatedAtGteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_gte'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCreatedAtGteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_gte'));
    }

    /**
     * Test property "created_at_gt"
     * @dataProvider validCreatedAtGtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCreatedAtGt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCreatedAtGt());
        self::assertEmpty($instance->created_at_gt);
        $instance->setCreatedAtGt($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCreatedAtGt());
            self::assertNotNull($instance->getCreatedAtGt());
            self::assertNotNull($instance->created_at_gt);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCreatedAtGt());
                self::assertEquals($value, $instance->created_at_gt);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCreatedAtGt());
                self::assertEquals(new Datetime($value), $instance->created_at_gt);
            }
        }
    }

    /**
     * Test invalid property "created_at_gt"
     * @dataProvider invalidCreatedAtGtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCreatedAtGt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCreatedAtGt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCreatedAtGtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_gt'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCreatedAtGtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_gt'));
    }

    /**
     * Test property "created_at_lte"
     * @dataProvider validCreatedAtLteDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCreatedAtLte(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCreatedAtLte());
        self::assertEmpty($instance->created_at_lte);
        $instance->setCreatedAtLte($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCreatedAtLte());
            self::assertNotNull($instance->getCreatedAtLte());
            self::assertNotNull($instance->created_at_lte);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCreatedAtLte());
                self::assertEquals($value, $instance->created_at_lte);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCreatedAtLte());
                self::assertEquals(new Datetime($value), $instance->created_at_lte);
            }
        }
    }

    /**
     * Test invalid property "created_at_lte"
     * @dataProvider invalidCreatedAtLteDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCreatedAtLte(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCreatedAtLte($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCreatedAtLteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_lte'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCreatedAtLteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_lte'));
    }

    /**
     * Test property "created_at_lt"
     * @dataProvider validCreatedAtLtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCreatedAtLt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCreatedAtLt());
        self::assertEmpty($instance->created_at_lt);
        $instance->setCreatedAtLt($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCreatedAtLt());
            self::assertNotNull($instance->getCreatedAtLt());
            self::assertNotNull($instance->created_at_lt);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCreatedAtLt());
                self::assertEquals($value, $instance->created_at_lt);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCreatedAtLt());
                self::assertEquals(new Datetime($value), $instance->created_at_lt);
            }
        }
    }

    /**
     * Test invalid property "created_at_lt"
     * @dataProvider invalidCreatedAtLtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCreatedAtLt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCreatedAtLt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCreatedAtLtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_lt'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCreatedAtLtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_lt'));
    }

    /**
     * Test property "expires_at_gte"
     * @dataProvider validExpiresAtGteDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExpiresAtGte(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getExpiresAtGte());
        self::assertEmpty($instance->expires_at_gte);
        $instance->setExpiresAtGte($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasExpiresAtGte());
            self::assertNotNull($instance->getExpiresAtGte());
            self::assertNotNull($instance->expires_at_gte);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getExpiresAtGte());
                self::assertEquals($value, $instance->expires_at_gte);
            } else {
                self::assertEquals(new Datetime($value), $instance->getExpiresAtGte());
                self::assertEquals(new Datetime($value), $instance->expires_at_gte);
            }
        }
    }

    /**
     * Test invalid property "expires_at_gte"
     * @dataProvider invalidExpiresAtGteDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExpiresAtGte(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExpiresAtGte($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExpiresAtGteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at_gte'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExpiresAtGteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at_gte'));
    }

    /**
     * Test property "expires_at_gt"
     * @dataProvider validExpiresAtGtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExpiresAtGt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getExpiresAtGt());
        self::assertEmpty($instance->expires_at_gt);
        $instance->setExpiresAtGt($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasExpiresAtGt());
            self::assertNotNull($instance->getExpiresAtGt());
            self::assertNotNull($instance->expires_at_gt);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getExpiresAtGt());
                self::assertEquals($value, $instance->expires_at_gt);
            } else {
                self::assertEquals(new Datetime($value), $instance->getExpiresAtGt());
                self::assertEquals(new Datetime($value), $instance->expires_at_gt);
            }
        }
    }

    /**
     * Test invalid property "expires_at_gt"
     * @dataProvider invalidExpiresAtGtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExpiresAtGt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExpiresAtGt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExpiresAtGtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at_gt'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExpiresAtGtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at_gt'));
    }

    /**
     * Test property "expires_at_lte"
     * @dataProvider validExpiresAtLteDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExpiresAtLte(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getExpiresAtLte());
        self::assertEmpty($instance->expires_at_lte);
        $instance->setExpiresAtLte($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasExpiresAtLte());
            self::assertNotNull($instance->getExpiresAtLte());
            self::assertNotNull($instance->expires_at_lte);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getExpiresAtLte());
                self::assertEquals($value, $instance->expires_at_lte);
            } else {
                self::assertEquals(new Datetime($value), $instance->getExpiresAtLte());
                self::assertEquals(new Datetime($value), $instance->expires_at_lte);
            }
        }
    }

    /**
     * Test invalid property "expires_at_lte"
     * @dataProvider invalidExpiresAtLteDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExpiresAtLte(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExpiresAtLte($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExpiresAtLteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at_lte'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExpiresAtLteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at_lte'));
    }

    /**
     * Test property "expires_at_lt"
     * @dataProvider validExpiresAtLtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExpiresAtLt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getExpiresAtLt());
        self::assertEmpty($instance->expires_at_lt);
        $instance->setExpiresAtLt($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasExpiresAtLt());
            self::assertNotNull($instance->getExpiresAtLt());
            self::assertNotNull($instance->expires_at_lt);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getExpiresAtLt());
                self::assertEquals($value, $instance->expires_at_lt);
            } else {
                self::assertEquals(new Datetime($value), $instance->getExpiresAtLt());
                self::assertEquals(new Datetime($value), $instance->expires_at_lt);
            }
        }
    }

    /**
     * Test invalid property "expires_at_lt"
     * @dataProvider invalidExpiresAtLtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExpiresAtLt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExpiresAtLt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExpiresAtLtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at_lt'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExpiresAtLtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at_lt'));
    }

    /**
     * Test property "status"
     * @dataProvider validStatusDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testStatus(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getStatus());
        self::assertEmpty($instance->status);
        $instance->setStatus($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasStatus());
            self::assertNotNull($instance->getStatus());
            self::assertNotNull($instance->status);
            self::assertEquals($value, is_array($value) ? $instance->getStatus()->toArray() : $instance->getStatus());
            self::assertEquals($value, is_array($value) ? $instance->status->toArray() : $instance->status);
            self::assertContains($instance->getStatus(), ['opened', 'closed']);
            self::assertContains($instance->status, ['opened', 'closed']);
        }
    }

    /**
     * Test invalid property "status"
     * @dataProvider invalidStatusDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidStatus(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setStatus($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validStatusDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_status'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidStatusDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_status'));
    }

    /**
     * Test property "full_text_search"
     * @dataProvider validFullTextSearchDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testFullTextSearch(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getFullTextSearch());
        self::assertEmpty($instance->full_text_search);
        $instance->setFullTextSearch($value);
        self::assertEquals($value, is_array($value) ? $instance->getFullTextSearch()->toArray() : $instance->getFullTextSearch());
        self::assertEquals($value, is_array($value) ? $instance->full_text_search->toArray() : $instance->full_text_search);
        if (!empty($value)) {
            self::assertTrue($instance->hasFullTextSearch());
            self::assertNotNull($instance->getFullTextSearch());
            self::assertNotNull($instance->full_text_search);
            self::assertLessThanOrEqual(128, is_string($instance->getFullTextSearch()) ? mb_strlen($instance->getFullTextSearch()) : $instance->getDescription());
            self::assertLessThanOrEqual(128, is_string($instance->full_text_search) ? mb_strlen($instance->full_text_search) : $instance->full_text_search);
        }
    }

    /**
     * Test invalid property "full_text_search"
     * @dataProvider invalidFullTextSearchDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidFullTextSearch(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setFullTextSearch($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validFullTextSearchDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_full_text_search'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidFullTextSearchDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_full_text_search'));
    }

    /**
     * Test valid method "builder"
     * @dataProvider validClassDataProvider
     * @param mixed $value
     *
     * @return void
     */
    public function testBuilder(mixed $value): void
    {
        $instance = $this->getTestInstance($value);
        self::assertInstanceOf(AbstractRequestBuilder::class, $instance::builder());
    }

    /**
     * @return array
     * @throws Exception
     */
    public function validClassDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return [$this->getValidDataProviderByClass($instance)];
    }
}
