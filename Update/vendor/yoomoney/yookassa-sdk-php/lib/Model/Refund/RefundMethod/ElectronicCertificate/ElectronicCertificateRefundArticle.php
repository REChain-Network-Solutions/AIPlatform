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

namespace YooKassa\Model\Refund\RefundMethod\ElectronicCertificate;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель ElectronicCertificateRefundArticle.
 *
 * Товарная позиция в корзине возврата при возврате на электронный сертификат.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property int $article_number Порядковый номер товара в корзине возврата. От 1 до 999 включительно.
 * @property int $articleNumber Порядковый номер товара в корзине возврата. От 1 до 999 включительно.
 * @property int $payment_article_number Порядковый номер товара в одобренной корзине покупки (article_number в [объекте платежа](https://yookassa.ru/developers/api#payment_object)). От 1 до 999 включительно.
 * @property int $paymentArticleNumber Порядковый номер товара в одобренной корзине покупки (article_number в [объекте платежа](https://yookassa.ru/developers/api#payment_object)). От 1 до 999 включительно.
 * @property string $tru_code Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code)
 * @property string $truCode Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code)
 * @property int $quantity Количество возвращаемых единиц товара. Формат: целое положительное число.
 */
class ElectronicCertificateRefundArticle extends AbstractObject
{
    /**
     * Порядковый номер товара в корзине возврата. От 1 до 999 включительно.
     *
     * @var int|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    private ?int $_article_number = null;

    /**
     * Порядковый номер товара в одобренной корзине покупки (article_number в [объекте платежа](https://yookassa.ru/developers/api#payment_object)). От 1 до 999 включительно.
     *
     * @var int|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    private ?int $_payment_article_number = null;

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
     * Количество возвращаемых единиц товара. Формат: целое положительное число.
     *
     * @var int|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(1)]
    private ?int $_quantity = null;

    /**
     * Возвращает article_number.
     *
     * @return int|null Порядковый номер товара в корзине возврата
     */
    public function getArticleNumber(): ?int
    {
        return $this->_article_number;
    }

    /**
     * Устанавливает article_number.
     *
     * @param int|null $article_number Порядковый номер товара в корзине возврата
     *
     * @return self
     */
    public function setArticleNumber(?int $article_number = null): self
    {
        $this->_article_number = $this->validatePropertyValue('_article_number', $article_number);
        return $this;
    }

    /**
     * Возвращает payment_article_number.
     *
     * @return int|null Порядковый номер товара в одобренной корзине покупки
     */
    public function getPaymentArticleNumber(): ?int
    {
        return $this->_payment_article_number;
    }

    /**
     * Устанавливает payment_article_number.
     *
     * @param int|null $payment_article_number Порядковый номер товара в одобренной корзине покупки
     *
     * @return self
     */
    public function setPaymentArticleNumber(?int $payment_article_number = null): self
    {
        $this->_payment_article_number = $this->validatePropertyValue('_payment_article_number', $payment_article_number);
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
     * @param string|null $tru_code Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate#payments-preparation-tru-code)
     *
     * @return self
     */
    public function setTruCode(?string $tru_code = null): self
    {
        $this->_tru_code = $this->validatePropertyValue('_tru_code', $tru_code);
        return $this;
    }

    /**
     * Возвращает quantity.
     *
     * @return int|null Количество возвращаемых единиц товара
     */
    public function getQuantity(): ?int
    {
        return $this->_quantity;
    }

    /**
     * Устанавливает quantity.
     *
     * @param int|null $quantity Количество возвращаемых единиц товара. Формат: целое положительное число.
     *
     * @return self
     */
    public function setQuantity(?int $quantity = null): self
    {
        $this->_quantity = $this->validatePropertyValue('_quantity', $quantity);
        return $this;
    }

}

