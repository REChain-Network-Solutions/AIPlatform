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

namespace YooKassa\Request\Payments;

use DateTime;
use Exception;
use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;

/**
 * Класс, представляющий модель PaymentsRequestBuilder.
 *
 * Класс билдера объекта запроса для получения списка платежей магазина, передаваемого в методы клиента API.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class PaymentsRequestBuilder extends AbstractRequestBuilder
{
    /**
     * @var PaymentsRequest|null Собираемый объект запроса списка платежей магазина
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Устанавливает страница выдачи результатов.
     *
     * @param null|string $value Страница выдачи результатов или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если в метод была передана не строка
     */
    public function setCursor(?string $value): PaymentsRequestBuilder
    {
        $this->currentObject->setCursor($value);

        return $this;
    }

    /**
     * Устанавливает дату создания от которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время создания, от (не включая) или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана
     *                                       строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не
     *                                                     строка, не число и не значение типа DateTime)
     */
    public function setCreatedAtGt(mixed $value): PaymentsRequestBuilder
    {
        $this->currentObject->setCreatedAtGt($value);

        return $this;
    }

    /**
     * Устанавливает дату создания от которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время создания, от (включительно) или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана
     *                                       строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не
     *                                                     строка, не число и не значение типа DateTime)
     */
    public function setCreatedAtGte(mixed $value): PaymentsRequestBuilder
    {
        $this->currentObject->setCreatedAtGte($value);

        return $this;
    }

    /**
     * Устанавливает дату создания до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время создания, до (не включая) или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана
     *                                       строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не
     *                                                     строка, не число и не значение типа DateTime)
     */
    public function setCreatedAtLt(mixed $value): PaymentsRequestBuilder
    {
        $this->currentObject->setCreatedAtLt($value);

        return $this;
    }

    /**
     * Устанавливает дату создания до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время создания, до (включительно) или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана
     *                                       строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не
     *                                                     строка, не число и не значение типа DateTime)
     */
    public function setCreatedAtLte(mixed $value): PaymentsRequestBuilder
    {
        $this->currentObject->setCreatedAtLte($value);

        return $this;
    }

    /**
     * Устанавливает дату подтверждения от которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время создания, до (включительно) или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана
     *                                       строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не
     *                                                     строка, не число и не значение типа DateTime)
     */
    public function setCapturedAtGt(mixed $value): PaymentsRequestBuilder
    {
        $this->currentObject->setCapturedAtGt($value);

        return $this;
    }

    /**
     * Устанавливает дату подтверждения от которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время подтверждения, от (включительно) или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана
     *                                       строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не
     *                                                     строка, не число и не значение типа DateTime)
     */
    public function setCapturedAtGte(mixed $value): PaymentsRequestBuilder
    {
        $this->currentObject->setCapturedAtGte($value);

        return $this;
    }

    /**
     * Устанавливает дату подтверждения до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время подтверждения, до (включительно) или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана
     *                                       строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не
     *                                                     строка, не число и не значение типа DateTime)
     */
    public function setCapturedAtLt(mixed $value): PaymentsRequestBuilder
    {
        $this->currentObject->setCapturedAtLt($value);

        return $this;
    }

    /**
     * Устанавливает дату подтверждения до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $value Время подтверждения, до (включительно) или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Генерируется если была передана дата в невалидном формате (была передана
     *                                       строка или число, которые не удалось преобразовать в валидную дату)
     * @throws Exception|InvalidPropertyValueTypeException Генерируется если была передана дата с не тем типом (передана не
     *                                                     строка, не число и не значение типа DateTime)
     */
    public function setCapturedAtLte(mixed $value): PaymentsRequestBuilder
    {
        $this->currentObject->setCapturedAtLte($value);

        return $this;
    }

    /**
     * Устанавливает ограничение количества объектов платежа.
     *
     * @param string|int|null $value Ограничение количества объектов платежа или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если в метод было передана не целое число
     */
    public function setLimit(mixed $value): PaymentsRequestBuilder
    {
        $this->currentObject->setLimit($value);

        return $this;
    }

    /**
     * Устанавливает статус выбираемых платежей.
     *
     * @param string|null $value Статус выбираемых платежей или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Выбрасывается если переданное значение не является валидным статусом
     * @throws InvalidPropertyValueTypeException Выбрасывается если в метод была передана не строка
     */
    public function setStatus(?string $value): PaymentsRequestBuilder
    {
        $this->currentObject->setStatus($value);

        return $this;
    }

    /**
     * Устанавливает платежный метод выбираемых платежей.
     *
     * @param string|null $value Платежный метод выбираемых платежей или null, чтобы удалить значение
     *
     * @return PaymentsRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueException Выбрасывается если переданное значение не является валидным статусом
     * @throws InvalidPropertyValueTypeException Выбрасывается если в метод была передана не строка
     */
    public function setPaymentMethod(?string $value): PaymentsRequestBuilder
    {
        $this->currentObject->setPaymentMethod($value);

        return $this;
    }

    /**
     * Собирает и возвращает объект запроса списка платежей магазина.
     *
     * @param null|array $options Массив с настройками запроса
     *
     * @return PaymentsRequest Инстанс объекта запроса к API для получения списка платежей магазина
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        return parent::build($options);
    }

    /**
     * Возвращает новый объект запроса для получения списка платежей, который в дальнейшем будет собираться в билдере.
     *
     * @return PaymentsRequest Объект запроса списка платежей магазина
     */
    protected function initCurrentObject(): AbstractRequestInterface
    {
        return new PaymentsRequest();
    }
}
