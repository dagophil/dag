#include <iostream>

#include <vigra/jungle.hxx>



template <typename JUNGLE>
void test_jungle()
{
    using namespace vigra;

    typedef JUNGLE Jungle;

    Jungle g;

}


int main()
{
    test_jungle<vigra::ListRandomAccessJungle>();
}
