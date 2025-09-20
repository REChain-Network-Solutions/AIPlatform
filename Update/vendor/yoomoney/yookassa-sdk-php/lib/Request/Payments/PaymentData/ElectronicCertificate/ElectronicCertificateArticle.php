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

namespace YooKassa\Request\Payments\PaymentData\ElectronicCertificate;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель ElectronicCertificateArticle.
 *
 * Товарная позиция в корзине покупки при оплате по электронному сертификату.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property int $article_number Порядковый номер товара в корзине. От 1 до 999 включительно.
 * @property int $articleNumber Порядковый номер товара в корзине. От 1 до 999 включительно.
 * @property string $tru_code Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code)
 * @property string $truCode Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code)
 * @property string $article_code Код товара в вашей системе. Максимум 128 символов.
 * @property string $articleCode Код товара в вашей системе. Максимум 128 символов.
 * @property string $article_name Название товара в вашей системе. Отображается на готовой платежной форме ЮKassa. Максимум 128 символов.
 * @property string $articleName Название товара в вашей системе. Отображается на готовой платежной форме ЮKassa. Максимум 128 символов.
 * @property int $quantity Количество единиц товара. Формат: целое положительное число.
 * @property AmountInterface $price Цена за единицу товара.
 * @property Metadata $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
 */
class ElectronicCertificateArticle extends AbstractObject
{
    /**
     * Порядковый номер товара в корзине. От 1 до 999 включительно.
     *
     * @var int|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(1)]
    #[Assert\LessThanOrEqual(999)]
    private ?int $_article_number = null;

    /**
     * Код ТРУ. 30 символов, две группы цифр, разделенные точкой.
     *
     * Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя.
     *
     * Пример: ~`329921120.06001010200080001643`
     *
     * [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code)
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 30)]
    #[Assert\Length(min: 30)]
    private ?string $_tru_code = null;

    /**
     * Код товара в вашей системе. Максимум 128 символов.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 128)]
    private ?string $_article_code = null;

    /**
     * Название товара в вашей системе. Отображается на готовой платежной форме ЮKassa. Максимум 128 символов.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 128)]
    private ?string $_article_name = null;

    /**
     * Количество единиц товара. Формат: целое положительное число.
     *
     * @var int|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    private ?int $_quantity = null;

    /**
     * Цена за единицу товара.
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_price = null;

    /**
     * Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @var Metadata|null
     */
    #[Assert\AllType('string')]
    #[Assert\Type(Metadata::class)]
    private ?Metadata $_metadata = null;

    /**
     * Возвращает article_number.
     *
     * @return int|null Порядковый номер товара в корзине
     */
    public function getArticleNumber(): ?int
    {
        return $this->_article_number;
    }

    /**
     * Устанавливает article_number.
     *
     * @param int|null $article_number Порядковый номер товара в корзине. От 1 до 999 включительно.
     *
     * @return self
     */
    public function setArticleNumber(?int $article_number = null): self
    {
        $this->_article_number = $this->validatePropertyValue('_article_number', $article_number);
        return $this;
    }

    /**
     * Возвращает tru_code.
     *
     * @return string|null Код ТРУ
     */
    public function getTruCode(): ?string
    {
        return $this->_tru_code;
    }

    /**
     * Устанавливает tru_code.
     *
     * @param string|null $tru_code Код ТРУ. 30 символов
     *
     * @return self
     */
    public function setTruCode(?string $tru_code = null): self
    {
        $this->_tru_code = $this->validatePropertyValue('_tru_code', $tru_code);
        return $this;
    }

    /**
     * Возвращает article_code.
     *
     * @return string|null Код товара в вашей системе
     */
    public function getArticleCode(): ?string
    {
        return $this->_article_code;
    }

    /**
     * Устанавливает article_code.
     *
     * @param string|null $article_code Код товара в вашей системе. Максимум 128 символов.
     *
     * @return self
     */
    public function setArticleCode(?string $article_code = null): self
    {
        $this->_article_code = $this->validatePropertyValue('_article_code', $article_code);
        return $this;
    }

    /**
     * Возвращает article_name.
     *
     * @return string|null Название товара в вашей системе
     */
    public function getArticleName(): ?string
    {
        return $this->_article_name;
    }

    /**
     * Устанавливает article_name.
     *
     * @param string|null $article_name Название товара в вашей системе. Отображается на готовой платежной форме ЮKassa. Максимум 128 символов.
     *
     * @return self
     */
    public function setArticleName(?string $article_name = null): self
    {
        $this->_article_name = $this->validatePropertyValue('_article_name', $article_name);
        return $this;
    }

    /**
     * Возвращает quantity.
     *
     * @return int|null Количество единиц товара
     */
    public function getQuantity(): ?int
    {
        return $this->_quantity;
    }

    /**
     * Устанавливает quantity.
     *
     * @param int|null $quantity Количество единиц товара. Формат: целое положительное число.
     *
     * @return self
     */
    public function setQuantity(?int $quantity = null): self
    {
        $this->_quantity = $this->validatePropertyValue('_quantity', $quantity);
        return $this;
    }

    /**
     * Возвращает price.
     *
     * @return AmountInterface|null Цена за единицу товара
     */
    public function getPrice(): ?AmountInterface
    {
        return $this->_price;
    }

    /**
     * Устанавливает price.
     *
     * @param AmountInterface|array|null $price Цена за единицу товара.
     *
     * @return self
     */
    public function setPrice(mixed $price = null): self
    {
        $this->_price = $this->validatePropertyValue('_price', $price);
        return $this;
    }

    /**
     * Возвращает metadata.
     *
     * @return Metadata|null Любые дополнительные данные
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Устанавливает metadata.
     *
     * @param Metadata|array|null $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @return self
     */
    public function setMetadata(mixed $metadata = null): self
    {
        $this->_metadata = $this->validatePropertyValue('_metadata', $metadata);
        return $this;
    }

}

