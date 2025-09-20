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

namespace YooKassa\Model\Invoice;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель LineItem.
 *
 * Данные о товаре или услуге в корзине.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $description Название товара или услуги (от 1 до 128 символов). Пользователь увидит его на странице счета перед оплатой.
 * @property AmountInterface $price Полная цена товара или услуги. Пользователь увидит ее на странице счета перед оплатой.
 * @property AmountInterface|null $discount_price Итоговая цена товара с учетом скидки. Если передана, то на странице счета цена отобразится с учетом скидки. Не нужно передавать, если пользователь оплачивает полную стоимость товара или услуги.
 * @property AmountInterface|null $discountPrice Итоговая цена товара с учетом скидки. Если передана, то на странице счета цена отобразится с учетом скидки. Не нужно передавать, если пользователь оплачивает полную стоимость товара или услуги.
 * @property float $quantity Количество товара. Можно передать целое или дробное число. Разделитель дробной части — точка, разделитель тысяч отсутствует, максимум три знака после точки. Пример: ~`5.000`
 */
class LineItem extends AbstractObject
{
    /** @var int Минимальная длина название товара или услуги */
    public const MIN_LENGTH_DESCRIPTION = 1;

    /** @var int Максимальная длина название товара или услуги */
    public const MAX_LENGTH_DESCRIPTION = 128;

    /**
     * Название товара или услуги (от 1 до 128 символов). Пользователь увидит его на странице счета перед оплатой.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_DESCRIPTION)]
    #[Assert\Length(min: self::MIN_LENGTH_DESCRIPTION)]
    private ?string $_description = null;

    /**
     * Полная цена товара или услуги. Пользователь увидит ее на странице счета перед оплатой.
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_price = null;

    /**
     * Итоговая цена товара с учетом скидки. Если передана, то на странице счета цена отобразится с учетом скидки. Не нужно передавать, если пользователь оплачивает полную стоимость товара или услуги.
     *
     * @var AmountInterface|null
     */
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_discount_price = null;

    /**
     * Количество товара. Можно передать целое или дробное число. Разделитель дробной части — точка, разделитель тысяч отсутствует, максимум три знака после точки. Пример: ~`5.000`
     *
     * @var float|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('float')]
    #[Assert\GreaterThan(0)]
    private ?float $_quantity = null;

    /**
     * Возвращает description.
     *
     * @return string|null Название товара или услуги
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает description.
     *
     * @param string|null $description Название товара или услуги (от 1 до 128 символов). Пользователь увидит его на странице счета перед оплатой.
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Возвращает price.
     *
     * @return AmountInterface|null Полная цена товара или услуги
     */
    public function getPrice(): ?AmountInterface
    {
        return $this->_price;
    }

    /**
     * Устанавливает price.
     *
     * @param AmountInterface|array|null $price Полная цена товара или услуги
     *
     * @return self
     */
    public function setPrice(mixed $price = null): self
    {
        $this->_price = $this->validatePropertyValue('_price', $price);
        return $this;
    }

    /**
     * Возвращает discount_price.
     *
     * @return AmountInterface|null Итоговая цена товара с учетом скидки
     */
    public function getDiscountPrice(): ?AmountInterface
    {
        return $this->_discount_price;
    }

    /**
     * Устанавливает discount_price.
     *
     * @param AmountInterface|array|null $discount_price Итоговая цена товара с учетом скидки
     *
     * @return self
     */
    public function setDiscountPrice(mixed $discount_price = null): self
    {
        $this->_discount_price = $this->validatePropertyValue('_discount_price', $discount_price);
        return $this;
    }

    /**
     * Возвращает quantity.
     *
     * @return float|null Количество товара
     */
    public function getQuantity(): ?float
    {
        return $this->_quantity;
    }

    /**
     * Устанавливает quantity.
     *
     * @param float|null $quantity Количество товара. Можно передать целое или дробное число. Разделитель дробной части — точка, разделитель тысяч отсутствует, максимум три знака после точки. Пример: ~`5.000`
     *
     * @return self
     */
    public function setQuantity(?float $quantity = null): self
    {
        $this->_quantity = $this->validatePropertyValue('_quantity', $quantity);
        return $this;
    }

}

