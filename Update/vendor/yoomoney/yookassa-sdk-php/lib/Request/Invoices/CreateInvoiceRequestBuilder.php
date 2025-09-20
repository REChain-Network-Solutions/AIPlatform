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

namespace YooKassa\Request\Invoices;

use DateTime;
use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Invoice\LineItem;
use YooKassa\Model\Metadata;
use YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData;

/**
 * Класс, представляющий модель CreateInvoiceRequestBuilder.
 *
 * Класс билдера объекта запроса на создание платежа, передаваемого в методы клиента API.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @example 02-builder.php 11 75 Пример использования билдера
 */
class CreateInvoiceRequestBuilder extends AbstractRequestBuilder
{
    /**
     * Собираемый объект запроса.
     *
     * @var CreateInvoiceRequestInterface|null
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Устанавливает payment_data.
     *
     * @param PaymentData|array|null $value Данные для проведения платежа по выставленному счету
     *
     * @return self
     */
    public function setPaymentData(mixed $value = null): CreateInvoiceRequestBuilder
    {
        $this->currentObject->setPaymentData($value);

        return $this;
    }

    /**
     * Устанавливает cart.
     *
     * @param ListObjectInterface|array|null $value Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.
     *
     * @return self
     */
    public function setCart(mixed $value = null):  CreateInvoiceRequestBuilder
    {
        $this->currentObject->setCart($value);

        return $this;
    }

    /**
     * Добавляет cart_item.
     *
     * @param LineItem|array|null $value Товар или услуга.
     *
     * @return self
     */
    public function addCartItem(mixed $value = null):  CreateInvoiceRequestBuilder
    {
        $this->currentObject->getCart()->add($value);

        return $this;
    }

    /**
     * Устанавливает delivery_method_data.
     *
     * @param AbstractDeliveryMethodData|array|null $value Данные о способе доставки счета пользователю
     *
     * @return self
     */
    public function setDeliveryMethodData(mixed $value = null): CreateInvoiceRequestBuilder
    {
        $this->currentObject->setDeliveryMethodData($value);

        return $this;
    }

    /**
     * Устанавливает expires_at.
     *
     * @param DateTime|string|null $value Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z`
     *
     * @return self
     */
    public function setExpiresAt(DateTime|string|null $value = null): CreateInvoiceRequestBuilder
    {
        $this->currentObject->setExpiresAt($value);

        return $this;
    }

    /**
     * Устанавливает locale.
     *
     * @param string|null $value Язык интерфейса, писем и смс, которые будет видеть или получать пользователь
     *
     * @return self
     */
    public function setLocale(?string $value = null): CreateInvoiceRequestBuilder
    {
        $this->currentObject->setLocale($value);

        return $this;
    }

    /**
     * Устанавливает метаданные, привязанные к счету.
     *
     * @param null|array|Metadata $value Метаданные платежа, устанавливаемые мерчантом
     *
     * @return CreateInvoiceRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданные данные не удалось интерпретировать как
     *                                           метаданные платежа
     */
    public function setMetadata(mixed $value): CreateInvoiceRequestBuilder
    {
        $this->currentObject->setMetadata($value);

        return $this;
    }

    /**
     * Устанавливает описание транзакции.
     *
     * @param string|null $value Описание транзакции
     *
     * @return CreateInvoiceRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Выбрасывается если переданное значение превышает допустимую длину
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданное значение не является строкой
     */
    public function setDescription(?string $value): CreateInvoiceRequestBuilder
    {
        $this->currentObject->setDescription($value);

        return $this;
    }

    /**
     * Строит и возвращает объект запроса для отправки в API ЮKassa.
     *
     * @param null|array $options Массив параметров для установки в объект запроса
     *
     * @return CreateInvoiceRequestInterface|AbstractRequestInterface Инстанс объекта запроса
     *
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        if (!empty($options)) {
            $this->setOptions($options);
        }

        return parent::build();
    }

    /**
     * Инициализирует объект запроса, который в дальнейшем будет собираться билдером
     *
     * @return CreateInvoiceRequest Инстанс собираемого объекта запроса к API
     */
    protected function initCurrentObject(): AbstractRequestInterface
    {
        return new CreateInvoiceRequest();
    }

}
