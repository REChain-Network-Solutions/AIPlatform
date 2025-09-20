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

namespace Tests\YooKassa\Request\Payments\PaymentData\ElectronicCertificate;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle;

/**
 * ElectronicCertificateArticleTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class ElectronicCertificateArticleTest extends AbstractTestCase
{
    protected ElectronicCertificateArticle $object;

    /**
     * @return ElectronicCertificateArticle
     */
    protected function getTestInstance(): ElectronicCertificateArticle
    {
        return new ElectronicCertificateArticle();
    }

    /**
     * @return void
     */
    public function testElectronicCertificateArticleClassExists(): void
    {
        $this->object = $this->getMockBuilder(ElectronicCertificateArticle::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ElectronicCertificateArticle::class));
        $this->assertInstanceOf(ElectronicCertificateArticle::class, $this->object);
    }

    /**
     * Test property "article_number"
     * @dataProvider validArticleNumberDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testArticleNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setArticleNumber($value);
        self::assertNotNull($instance->getArticleNumber());
        self::assertNotNull($instance->article_number);
        self::assertEquals($value, $instance->getArticleNumber());
        self::assertEquals($value, $instance->article_number);
        self::assertLessThanOrEqual(999, is_string($instance->getArticleNumber()) ? mb_strlen($instance->getArticleNumber()) : $instance->getArticleNumber());
        self::assertLessThanOrEqual(999, is_string($instance->article_number) ? mb_strlen($instance->article_number) : $instance->article_number);
        self::assertGreaterThanOrEqual(1, is_string($instance->getArticleNumber()) ? mb_strlen($instance->getArticleNumber()) : $instance->getArticleNumber());
        self::assertGreaterThanOrEqual(1, is_string($instance->article_number) ? mb_strlen($instance->article_number) : $instance->article_number);
        self::assertIsNumeric($instance->getArticleNumber());
        self::assertIsNumeric($instance->article_number);
    }

    /**
     * Test invalid property "article_number"
     * @dataProvider invalidArticleNumberDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidArticleNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setArticleNumber($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validArticleNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_article_number'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidArticleNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_article_number'));
    }

    /**
     * Test property "tru_code"
     * @dataProvider validTruCodeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testTruCode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setTruCode($value);
        self::assertNotNull($instance->getTruCode());
        self::assertNotNull($instance->tru_code);
        self::assertEquals($value, $instance->getTruCode());
        self::assertEquals($value, $instance->tru_code);
        self::assertLessThanOrEqual(30, is_string($instance->getTruCode()) ? mb_strlen($instance->getTruCode()) : $instance->getTruCode());
        self::assertLessThanOrEqual(30, is_string($instance->tru_code) ? mb_strlen($instance->tru_code) : $instance->tru_code);
        self::assertGreaterThanOrEqual(30, is_string($instance->getTruCode()) ? mb_strlen($instance->getTruCode()) : $instance->getTruCode());
        self::assertGreaterThanOrEqual(30, is_string($instance->tru_code) ? mb_strlen($instance->tru_code) : $instance->tru_code);
    }

    /**
     * Test invalid property "tru_code"
     * @dataProvider invalidTruCodeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidTruCode(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setTruCode($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validTruCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_tru_code'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTruCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_tru_code'));
    }

    /**
     * Test property "article_code"
     * @dataProvider validArticleCodeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testArticleCode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getArticleCode());
        self::assertEmpty($instance->article_code);
        $instance->setArticleCode($value);
        self::assertEquals($value, $instance->getArticleCode());
        self::assertEquals($value, $instance->article_code);
        if (!empty($value)) {
            self::assertNotNull($instance->getArticleCode());
            self::assertNotNull($instance->article_code);
            self::assertLessThanOrEqual(128, is_string($instance->getArticleCode()) ? mb_strlen($instance->getArticleCode()) : $instance->getArticleCode());
            self::assertLessThanOrEqual(128, is_string($instance->article_code) ? mb_strlen($instance->article_code) : $instance->article_code);
        }
    }

    /**
     * Test invalid property "article_code"
     * @dataProvider invalidArticleCodeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidArticleCode(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setArticleCode($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validArticleCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_article_code'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidArticleCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_article_code'));
    }

    /**
     * Test property "article_name"
     * @dataProvider validArticleNameDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testArticleName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setArticleName($value);
        self::assertNotNull($instance->getArticleName());
        self::assertNotNull($instance->article_name);
        self::assertEquals($value, $instance->getArticleName());
        self::assertEquals($value, $instance->article_name);
        self::assertLessThanOrEqual(128, is_string($instance->getArticleName()) ? mb_strlen($instance->getArticleName()) : $instance->getArticleName());
        self::assertLessThanOrEqual(128, is_string($instance->article_name) ? mb_strlen($instance->article_name) : $instance->article_name);
    }

    /**
     * Test invalid property "article_name"
     * @dataProvider invalidArticleNameDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidArticleName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setArticleName($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validArticleNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_article_name'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidArticleNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_article_name'));
    }

    /**
     * Test property "quantity"
     * @dataProvider validQuantityDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testQuantity(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setQuantity($value);
        self::assertNotNull($instance->getQuantity());
        self::assertNotNull($instance->quantity);
        self::assertEquals($value, $instance->getQuantity());
        self::assertEquals($value, $instance->quantity);
        self::assertIsNumeric($instance->getQuantity());
        self::assertIsNumeric($instance->quantity);
    }

    /**
     * Test invalid property "quantity"
     * @dataProvider invalidQuantityDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidQuantity(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setQuantity($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_quantity'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_quantity'));
    }

    /**
     * Test property "price"
     * @dataProvider validPriceDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPrice(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setPrice($value);
        self::assertNotNull($instance->getPrice());
        self::assertNotNull($instance->price);
        self::assertEquals($value, is_array($value) ? $instance->getPrice()->toArray() : $instance->getPrice());
        self::assertEquals($value, is_array($value) ? $instance->price->toArray() : $instance->price);
    }

    /**
     * Test invalid property "price"
     * @dataProvider invalidPriceDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPrice(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPrice($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPriceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_price'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPriceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_price'));
    }

    /**
     * Test property "metadata"
     * @dataProvider validMetadataDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMetadata(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMetadata());
        self::assertEmpty($instance->metadata);
        $instance->setMetadata($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getMetadata());
            self::assertNotNull($instance->metadata);
            foreach ($value as $key => $element) {
                if (!empty($element)) {
                    self::assertEquals($element, $instance->getMetadata()[$key]);
                    self::assertEquals($element, $instance->metadata[$key]);
                    self::assertIsObject($instance->getMetadata());
                    self::assertIsObject($instance->metadata);
                }
            }
            self::assertCount(count($value), $instance->getMetadata());
            self::assertCount(count($value), $instance->metadata);
            if ($instance->getMetadata() instanceof Metadata) {
                self::assertEquals($value, $instance->getMetadata()->toArray());
                self::assertEquals($value, $instance->metadata->toArray());
                self::assertCount(count($value), $instance->getMetadata());
                self::assertCount(count($value), $instance->metadata);
            }
        }
    }

    /**
     * Test invalid property "metadata"
     * @dataProvider invalidMetadataDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMetadata(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMetadata($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMetadataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_metadata'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMetadataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_metadata'));
    }
}
